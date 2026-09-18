#!/usr/bin/env python3
"""stream_load.py — N concurrent WebSocket streams against mynah-asr-server, paced at real time.

    python3 tools/bench/stream_load.py --streams 8 --clips samples/en/*.wav tests/audio/test_it.wav \
        [--host localhost] [--port 8090] [--lang auto] [--lookahead 3] [--frame-ms 100] \
        [--pace 1.0] [--repeat 1] [--json out.json] [--reference ref.json]

Standard library only, so it runs on a bare box. One PROCESS per stream (a thread cannot
pace many streams under the GIL). Each stream plays its clip in `--frame-ms` frames on a
monotonic schedule, records the send time of every frame and the arrival time of every
server frame, then closes and waits for `{"done":true}`.

Metrics, per utterance (definitions shared with .work/bench-harness-streaming.md):
  TTFP           first non-empty text delta received  -  first frame sent
  emission lag   delta received  -  send time of the frame that carried the last sample the
                 server had consumed when it produced the delta (from `audio_seconds`)
  finalization   `done` received  -  last frame sent
  paced          max lateness of the send schedule; a run whose lateness exceeds half a frame
                 could not hold 1x and its cadence percentiles are printed as DIAGNOSTIC only
  text identity  every stream that played the same clip must produce byte-identical text;
                 a mismatch invalidates the run whatever the timing says

Percentiles are nearest-rank over utterances (or over deltas, where stated). The tool never
prints a percentile it cannot support: fewer than 2 samples prints the samples.
"""
from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import platform
import socket
import struct
import subprocess
import sys
import threading
import time
import wave

# ---------------------------------------------------------------- websocket framing (client)

def ws_send(sock: socket.socket, opcode: int, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", n)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    sock.sendall(header + mask + masked)


def ws_recv(sock: socket.socket) -> tuple[int, bytes]:
    def rd(n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("connection closed")
            buf += part
        return buf
    h = rd(2)
    opcode = h[0] & 0x0F
    plen = h[1] & 0x7F
    if plen == 126:
        plen = struct.unpack(">H", rd(2))[0]
    elif plen == 127:
        plen = struct.unpack(">Q", rd(8))[0]
    return opcode, rd(plen) if plen else b""


def ws_connect(host: str, port: int, path: str, timeout: float) -> tuple[socket.socket | None, str]:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                  f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        part = sock.recv(4096)
        if not part:
            break
        resp += part
    status = resp.split(b"\r\n", 1)[0].decode(errors="replace")
    if " 101 " not in status:
        sock.close()
        return None, status or "no response"
    return sock, status

# ---------------------------------------------------------------- one stream (one process)

def load_pcm(path: str) -> bytes:
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono s16 WAV")
        return w.readframes(w.getnframes())


def run_utterance(a, clip: str, pcm: bytes) -> dict:
    frame_bytes = int(16000 * a.frame_ms / 1000) * 2
    n_frames = (len(pcm) + frame_bytes - 1) // frame_bytes
    out = {"clip": clip, "audio_s": len(pcm) / 32000.0, "frames": n_frames, "error": None,
           "status": None, "ttfp_ms": None, "lags_ms": [], "fin_ms": None, "text": "",
           "lang": None, "deltas": 0, "empty_deltas": 0, "max_late_ms": 0.0}
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
    try:
        sock, status = ws_connect(a.host, a.port, path, timeout=a.connect_timeout)
    except OSError as e:
        out["error"] = f"connect: {e}"
        return out
    out["status"] = status
    if sock is None:
        out["error"] = f"refused: {status}"
        return out

    sends: list[float] = []            # send completion time of frame i (monotonic)
    events: list[tuple[float, dict]] = []
    done_at: list[float] = []
    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                op, payload = ws_recv(sock)
                now = time.monotonic()
                if op == 0x8:
                    break
                if op == 0x1:
                    msg = json.loads(payload)
                    if msg.get("done"):
                        done_at.append(now)
                        out["lang"] = msg.get("language")
                        break
                    events.append((now, msg))
        except Exception as e:  # noqa: BLE001 — recorded, not raised, in a client thread
            if not stop.is_set():
                out["error"] = out["error"] or f"reader: {e}"

    sock.settimeout(a.done_timeout)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    period = a.frame_ms / 1000.0 / a.pace
    t0 = time.monotonic()
    try:
        for i in range(n_frames):
            due = t0 + i * period
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)
            else:
                out["max_late_ms"] = max(out["max_late_ms"], (now - due) * 1000.0)
            ws_send(sock, 0x2, pcm[i * frame_bytes:(i + 1) * frame_bytes])
            sends.append(time.monotonic())
        ws_send(sock, 0x8, b"")
        t_last = sends[-1]
        th.join(a.done_timeout)
        if th.is_alive():
            out["error"] = out["error"] or "timeout waiting for done"
    except OSError as e:
        out["error"] = f"send: {e}"
    finally:
        stop.set()
        try:
            sock.close()
        except OSError:
            pass

    texts = []
    first_text_at = None
    for t_recv, msg in events:
        text = msg.get("text", "")
        if text:
            texts.append(text)
            if first_text_at is None:
                first_text_at = t_recv
            aud = msg.get("audio_seconds")
            if aud is not None and sends:
                k = min(n_frames - 1, max(0, int((aud * 1000.0) / a.frame_ms + 0.999) - 1))
                if k < len(sends):
                    out["lags_ms"].append((t_recv - sends[k]) * 1000.0)
        else:
            out["empty_deltas"] += 1
    out["deltas"] = len(texts)
    out["text"] = "".join(texts)
    if first_text_at is not None and sends:
        out["ttfp_ms"] = (first_text_at - sends[0]) * 1000.0
    if done_at and sends:
        out["fin_ms"] = (done_at[0] - t_last) * 1000.0
    return out


def stream_main(a, idx: int, clips: list[str], q) -> None:
    pcms = {c: load_pcm(c) for c in clips}
    for r in range(a.repeat):
        clip = clips[(idx + r) % len(clips)]
        res = run_utterance(a, clip, pcms[clip])
        res["stream"] = idx
        res["rep"] = r
        q.put(res)

# ---------------------------------------------------------------- aggregation

def pct(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    k = max(1, int(round(p / 100.0 * len(ys) + 0.5)))   # nearest rank, 1-based
    return ys[min(len(ys), k) - 1]


def fmt_pct(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    if len(xs) < 2:
        return f"{xs[0]:.0f} (1 sample)"
    return f"{pct(xs, 50):.0f} / {pct(xs, 95):.0f}  (n={len(xs)})"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--clips", nargs="+", required=True, help="16 kHz mono s16 WAVs, cycled over streams")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--lookahead", default="3")
    ap.add_argument("--frame-ms", type=int, default=100)
    ap.add_argument("--pace", type=float, default=1.0, help="1.0 = real time; 2.0 = twice as fast")
    ap.add_argument("--repeat", type=int, default=1, help="utterances per stream, back to back (closed loop)")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--done-timeout", type=float, default=60.0)
    ap.add_argument("--json", help="write per-utterance results and the summary here")
    ap.add_argument("--reference", help="JSON {clip: expected_text}; mismatches fail the run")
    a = ap.parse_args()

    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=stream_main, args=(a, i, a.clips, q), daemon=True) for i in range(a.streams)]
    t_start = time.monotonic()
    for p in procs:
        p.start()
    results = []
    expected = a.streams * a.repeat
    deadline = time.monotonic() + a.repeat * (a.done_timeout + 300)
    while len(results) < expected and time.monotonic() < deadline:
        try:
            results.append(q.get(timeout=1.0))
        except Exception:  # noqa: BLE001 — queue.Empty, keep waiting while children live
            if not any(p.is_alive() for p in procs) and q.empty():
                break
    for p in procs:
        p.join(5)
    wall = time.monotonic() - t_start

    ok = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]
    refused = [r for r in errors if r["error"].startswith("refused")]
    ttfp = [r["ttfp_ms"] for r in ok if r["ttfp_ms"] is not None]
    lag_pooled = [x for r in ok for x in r["lags_ms"]]
    lag_utt_p95 = [pct(r["lags_ms"], 95) for r in ok if len(r["lags_ms"]) >= 2]
    fin = [r["fin_ms"] for r in ok if r["fin_ms"] is not None]
    max_late = max([r["max_late_ms"] for r in results], default=0.0)
    paced = max_late <= a.frame_ms / 2.0 and a.pace == 1.0
    audio_s = sum(r["audio_s"] for r in ok)

    # text identity: same clip -> same text across streams; optional reference
    by_clip: dict[str, set[str]] = {}
    for r in ok:
        by_clip.setdefault(r["clip"], set()).add(r["text"])
    identity_fail = {c: sorted(t) for c, t in by_clip.items() if len(t) > 1}
    ref_fail = {}
    if a.reference:
        ref = json.load(open(a.reference))
        for c, texts in by_clip.items():
            exp = ref.get(c) or ref.get(os.path.basename(c))
            if exp is not None and texts != {exp}:
                ref_fail[c] = {"expected": exp, "got": sorted(texts)}

    # INVALID: nothing can be claimed (text differs, results missing, nothing succeeded).
    # Refusals (a real 503) are a legitimate outcome and do not invalidate the run.
    hard_errors = [r for r in errors if r not in refused]
    if identity_fail or ref_fail or len(results) < expected or not ok:
        verdict = "INVALID"
    elif not paced:
        verdict = "DIAGNOSTIC"
    else:
        verdict = "MEASURED" + (f" ({len(hard_errors)} errors)" if hard_errors else "")
    print(f"stream_load  host={a.host}:{a.port}  streams={a.streams}  repeat={a.repeat}  frame={a.frame_ms}ms  "
          f"pace={a.pace}  lookahead={a.lookahead}  lang={a.lang}  tree={git_rev()}")
    print(f"  utterances {len(ok)}/{expected} ok, {len(errors)} errors ({len(refused)} refused), "
          f"audio {audio_s:.1f} s in {wall:.1f} s wall")
    print(f"  pacing: max lateness {max_late:.1f} ms -> {'PACED' if paced else 'NOT PACED (cadence numbers are diagnostic)'}")
    print(f"  TTFP            p50 / p95 ms : {fmt_pct(ttfp)}")
    print(f"  emission lag    p50 / p95 ms : {fmt_pct(lag_pooled)}   [pooled over deltas]")
    print(f"  emission lag    per-utt p95  : {fmt_pct(lag_utt_p95)}   [p50/p95 of per-utterance p95]")
    print(f"  finalization    p50 / p95 ms : {fmt_pct(fin)}")
    if identity_fail:
        print(f"  TEXT IDENTITY FAIL: {len(identity_fail)} clip(s) produced different texts across streams")
        for c, ts in identity_fail.items():
            print(f"    {c}: {ts}")
    if ref_fail:
        print(f"  REFERENCE FAIL: {len(ref_fail)} clip(s) differ from the reference")
    for r in errors[:5]:
        print(f"  error stream {r['stream']} rep {r['rep']}: {r['error']}")
    print(f"  verdict: {verdict}")

    if a.json:
        summary = {
            "manifest": {"host": a.host, "port": a.port, "streams": a.streams, "repeat": a.repeat,
                         "frame_ms": a.frame_ms, "pace": a.pace, "lookahead": a.lookahead, "lang": a.lang,
                         "clips": a.clips, "tree": git_rev(), "python": platform.python_version(),
                         "client_host": platform.node(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "summary": {"ok": len(ok), "expected": expected, "errors": len(errors), "refused": len(refused),
                        "wall_s": wall, "audio_s": audio_s, "paced": paced, "max_late_ms": max_late,
                        "ttfp_ms": {"p50": pct(ttfp, 50), "p95": pct(ttfp, 95)} if len(ttfp) >= 2 else None,
                        "lag_ms": {"p50": pct(lag_pooled, 50), "p95": pct(lag_pooled, 95)} if len(lag_pooled) >= 2 else None,
                        "fin_ms": {"p50": pct(fin, 50), "p95": pct(fin, 95)} if len(fin) >= 2 else None,
                        "identity_fail": identity_fail, "reference_fail": ref_fail, "verdict": verdict},
            "utterances": results,
        }
        with open(a.json, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"  -> {a.json}")
    return 0 if verdict != "INVALID" else 2


if __name__ == "__main__":
    sys.exit(main())
