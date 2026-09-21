#!/usr/bin/env python3
"""stream_load.py — concurrent WebSocket streams against mynah-asr-server, paced at real time.

    # WAVE: screening.  N streams at once, R utterances each.  May disqualify, never promotes.
    python3 tools/bench/stream_load.py --mode wave --streams 4 --repeat 2 \
        --clips samples/en/*.wav tests/audio/test_it.wav --port 8090

    # SOAK: the production gate.  Closed loop at fixed concurrency over a stratified bank.
    python3 tools/bench/stream_load.py --mode soak --streams 8 --duration 600 \
        --warmup 30 --window 60 --bank short,medium,long --seed 42 \
        --clips samples/*/*.wav tests/audio/*.wav --port 8090 --json soak.json

Standard library only, so it runs on a bare box with no venv.  One PROCESS per stream (a
thread cannot pace many streams under the GIL).  Each stream plays its clip in `--frame-ms`
frames on a monotonic schedule and records the send time of every frame, the arrival time
of every server frame, and the pacing lateness.

**Every metric is defined in `tools/bench/streaming_metrics.py`** and nothing is computed
here: two harnesses with two definitions of "emission lag p95" produce two numbers that
cannot be compared across hosts or commits (`ENGINEERING.md` §8).  Run that module's
known-answer self-test with `make test`.

Vocabulary (`ENGINEERING.md` §8): WAVE screens, SOAK promotes, DIAGNOSTIC never quotes a
number.  A run that could not hold 1x pacing prints its cadence values labelled DIAGNOSTIC
and is INVALID for the envelope, whatever the timing says.  A `503` before the upgrade is a
REJECTION, not an error: it is the admission ladder working, and it is counted separately.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streaming_metrics as M          # noqa: E402  (the single definition of every metric)

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


def ws_recv(sock: socket.socket) -> "tuple[int, bytes]":
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


def ws_connect(host: str, port: int, path: str, timeout: float):
    """(sock|None, status line, headers).  A refusal arrives as an HTTP status before the
    upgrade, by design (`docs/server.md`), so the client reads a status, not a reset."""
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
    head = resp.split(b"\r\n\r\n", 1)[0].decode(errors="replace").split("\r\n")
    status = head[0] if head else "no response"
    headers = {}
    for line in head[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    if " 101 " not in status:
        sock.close()
        return None, status or "no response", headers
    return sock, status, headers

# ------------------------------------------------------------------------------- the bank


def wav_audio_s(path: str) -> float:
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono s16 WAV")
        return w.getnframes() / float(w.getframerate())


def load_pcm(path: str) -> bytes:
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16 kHz mono s16 WAV")
        return w.readframes(w.getnframes())


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_transcripts(path):
    """clip -> human reference text.

    Accepts either a plain {clip: text} map or a bank manifest in the shape this repo
    already uses (samples/manifest.json: a "samples" list whose entries carry "file" and
    "text"). Keys are stored as given, and ALSO under the bare file name -- but only
    when that name belongs to one clip. Eleven languages of samples/manifest.json share
    `fleurs_1521.wav`; registering it would make every one of them score against
    whichever language the manifest listed last. Matching is by path suffix
    (streaming_metrics.reference_for), so `it/fleurs_1521.wav` still answers for
    `samples/it/fleurs_1521.wav` without any basename key at all."""
    with open(path) as f:
        doc = json.load(f)
    out, by_base = {}, {}
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        pairs = [(e.get("file"), e.get("text")) for e in doc["samples"]]
    elif isinstance(doc, dict):
        pairs = [(k, v) for k, v in doc.items() if isinstance(v, str)]
    else:
        pairs = None
    if pairs is not None:
        for fn, txt in pairs:
            if fn and txt:
                out[fn] = txt
                by_base.setdefault(os.path.basename(fn), set()).add(txt)
        for base, texts in by_base.items():
            if len(texts) == 1 and base not in out:
                out[base] = next(iter(texts))
    else:
        raise SystemExit(f"--transcripts {path}: expected a manifest or a {{clip: text}} map")
    if not out:
        raise SystemExit(f"--transcripts {path}: no clip carried a reference text")
    return out


def classify(duration_s: float, bounds) -> str:
    """Duration classes for the stratified bank.  Boundaries are a flag, not a belief."""
    return "short" if duration_s < bounds[0] else ("medium" if duration_s < bounds[1] else "long")


def build_bank(clips, classes, bounds):
    """{class: [(clip, duration_s), ...]} for the requested classes only, sorted by name."""
    bank = {c: [] for c in classes}
    for c in sorted(set(clips)):
        d = wav_audio_s(c)
        k = classify(d, bounds)
        if k in bank:
            bank[k].append((c, d))
    empty = [k for k, v in bank.items() if not v]
    if empty:
        raise SystemExit(f"bank class(es) {','.join(empty)} have no clip in --clips "
                         f"(bounds {bounds[0]}/{bounds[1]} s); pass more clips or drop the class")
    return bank


def stratified_schedule(bank, classes, seed, n):
    """A deterministic, class-balanced sequence of clips.

    Round-robin over the classes so that every window of `len(classes)` utterances holds one
    of each; the clip inside a class is drawn from a seeded shuffle, so two runs with the
    same seed and the same bank play exactly the same audio in the same order."""
    rng = random.Random(seed)
    pools = {k: list(v) for k, v in bank.items()}
    for k in pools:
        rng.shuffle(pools[k])
    cursor = {k: 0 for k in pools}
    order, i = [], 0
    while len(order) < n:
        k = classes[i % len(classes)]
        i += 1
        pool = pools[k]
        clip = pool[cursor[k] % len(pool)][0]
        cursor[k] += 1
        order.append(clip)
    return order

# ---------------------------------------------------------------- one stream (one process)


def run_utterance(a, clip: str, pcm: bytes, cls: str) -> dict:
    """One utterance, recorded in the input format of `streaming_metrics`.

    Nothing is aggregated here: the record is raw marks.  `time.monotonic()` is system-wide
    on Linux (CLOCK_MONOTONIC) and macOS (mach_absolute_time), so marks taken in different
    stream processes share one timeline and can be windowed together by the parent."""
    frame_bytes = int(16000 * a.frame_ms / 1000) * 2
    n_frames = max(1, (len(pcm) + frame_bytes - 1) // frame_bytes)
    rec = {"clip": clip, "class": cls, "audio_s": len(pcm) / 32000.0, "sends": [], "late_ms": [],
           "events": [], "done_t": None, "t_start": time.monotonic(), "error": None,
           "rejected": False, "status": None, "lang": None, "retry_after": None}
    # `model=` names a WORKER GROUP in a multi-model fleet (S2-6). Sent only when
    # asked for, so a single-model server sees exactly the query it always saw.
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
    if a.model:
        path += f"&model={a.model}"
    try:
        sock, status, headers = ws_connect(a.host, a.port, path, timeout=a.connect_timeout)
    except OSError as e:
        rec["error"] = f"connect: {e}"
        return rec
    rec["status"] = status
    if sock is None:
        # a refusal before the upgrade is the admission ladder working, not a failure
        rec["rejected"] = " 503 " in status
        rec["retry_after"] = headers.get("retry-after")
        rec["error"] = f"refused: {status}"
        return rec

    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                op, payload = ws_recv(sock)
                now = time.monotonic()
                if op == 0x8:
                    break
                if op != 0x1:
                    continue
                msg = json.loads(payload)
                # v2 carries `type`/`seq`/`audio_s`/`lag_ms`; v1 carries `text`/`audio_seconds`
                # and `{"done":true}`.  Both are normalised to the metrics module's event.
                kind = msg.get("type")
                if kind is None:
                    kind = "done" if msg.get("done") else "delta"
                aud = msg.get("audio_s", msg.get("audio_seconds"))
                ev = {"t": now, "type": kind, "text": msg.get("text") or "",
                      "audio_s": None if aud is None else float(aud),
                      "lag_ms": None if msg.get("lag_ms") is None else float(msg["lag_ms"])}
                rec["events"].append(ev)
                if kind == "done":
                    rec["done_t"] = now
                    rec["lang"] = msg.get("lang") or msg.get("language")
                    break
                if kind == "error":
                    rec["error"] = rec["error"] or f"server error: {msg.get('code')} {msg.get('message')}"
                    break
        except Exception as e:                       # noqa: BLE001 — recorded, not raised
            if not stop.is_set():
                rec["error"] = rec["error"] or f"reader: {e}"

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
                time.sleep(due - now)              # the schedule, never a spin
            ws_send(sock, 0x2, pcm[i * frame_bytes:(i + 1) * frame_bytes])
            t_sent = time.monotonic()
            rec["sends"].append([t_sent, min((i + 1) * frame_bytes, len(pcm)) / 32000.0])
            rec["late_ms"].append(max(0.0, (t_sent - due) * 1000.0))
        ws_send(sock, 0x8, b"")                    # close = finalize, then `done`
        th.join(a.done_timeout)
        if th.is_alive():
            rec["error"] = rec["error"] or "timeout waiting for done"
    except OSError as e:
        rec["error"] = f"send: {e}"
    finally:
        stop.set()
        try:
            sock.close()
        except OSError:
            pass
    return rec


def stream_main(a, idx: int, schedule, classes_of, q) -> None:
    """One stream process: WAVE plays `--repeat` utterances, SOAK loops until the deadline.

    The schedule is the parent's; this process only walks its own stride of it, so the audio
    a given stream played is reproducible from (seed, bank, streams, index)."""
    pcms = {}
    deadline = a._t0 + a.duration if a.mode == "soak" else None
    k = 0
    while True:
        if a.mode == "wave" and k >= a.repeat:
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        clip = schedule[(idx + k * a.streams) % len(schedule)]
        if clip not in pcms:
            pcms[clip] = load_pcm(clip)
        rec = run_utterance(a, clip, pcms[clip], classes_of.get(clip, "n/a"))
        rec["stream"] = idx
        rec["rep"] = k
        q.put(rec)
        k += 1
        if rec["rejected"]:
            # honour Retry-After rather than hammering a full fleet (never a busy wait)
            try:
                back = float(rec["retry_after"])
            except (TypeError, ValueError):
                back = 0.25
            time.sleep(min(max(back, 0.05), 5.0))

# ------------------------------------------------------------------------------- manifest


def git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True,
                                             stderr=subprocess.DEVNULL).strip())
        return {"head": rev, "dirty": dirty}
    except Exception:                               # noqa: BLE001
        return {"head": "unknown", "dirty": None}


def health(host, port, timeout=2.0):
    """/v1/health if it answers; an unreachable endpoint is recorded as such, not invented."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/health", timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:                          # noqa: BLE001
        return {"unreachable": str(e)}

# ----------------------------------------------------------------------------------- main


def build_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--streams", type=int, default=1, help="concurrency (streams in flight)")
    ap.add_argument("--clips", nargs="+", required=True, help="16 kHz mono s16 WAVs")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--model", default=None,
                    help="the worker group to stream to, in a multi-model fleet "
                         "(?model=); omitted, the server's default group answers")
    ap.add_argument("--lookahead", default="3")
    ap.add_argument("--frame-ms", type=int, default=100)
    ap.add_argument("--pace", type=float, default=1.0, help="1.0 = real time; 2.0 = twice as fast")
    ap.add_argument("--repeat", type=int, default=1, help="WAVE: utterances per stream, back to back")
    ap.add_argument("--connect-timeout", type=float, default=10.0)
    ap.add_argument("--done-timeout", type=float, default=60.0)
    ap.add_argument("--json", help="write the manifest, the summary and every utterance here")
    ap.add_argument("--reference", help="JSON {clip: expected_text}; mismatches invalidate the run")
    ap.add_argument("--onsets", help="JSON {clip: speech_onset_seconds} from "
                                     "tools/bench/clip_onset.py: makes TTFP measurable "
                                     "from the moment speech begins, not from the moment "
                                     "the socket opened")
    ap.add_argument("--transcripts", help="bank manifest (samples/*/manifest.json) or a JSON "
                                          "{clip: human_reference}; gives every utterance a CER "
                                          "and lets the run be judged DEGRADED — cadence held, "
                                          "transcripts got worse")

    ap.add_argument("--mode", choices=("wave", "soak"), default="wave",
                    help="wave: N at once, R each, screening only. soak: closed loop, the gate")
    ap.add_argument("--duration", type=float, default=600.0, help="SOAK: seconds, warm-up included")
    ap.add_argument("--warmup", type=float, default=30.0, help="SOAK: leading seconds excluded from KPIs")
    ap.add_argument("--window", type=float, default=60.0, help="SOAK: analysis window, seconds")
    ap.add_argument("--bank", default="short,medium,long",
                    help="SOAK: duration classes to stratify over, comma-separated")
    ap.add_argument("--class-bounds", default="8,20",
                    help="seconds: short < b0 <= medium < b1 <= long")
    ap.add_argument("--seed", type=int, default=42, help="SOAK: schedule seed")

    ap.add_argument("--chunk-ms", type=float, default=None,
                    help="server encoder chunk period; default 80 x (lookahead+1) ms")
    ap.add_argument("--ttfp-p95-ms", type=float, default=None, help="envelope: default chunk + 200")
    ap.add_argument("--lag-p95-ms", type=float, default=None, help="envelope: default 1 chunk")
    ap.add_argument("--fin-p95-ms", type=float, default=500.0)
    ap.add_argument("--backlog-max-s", type=float, default=None, help="envelope: default 2 chunks")
    ap.add_argument("--max-drift-pct", type=float, default=20.0)
    ap.add_argument("--marginal-factor", type=float, default=1.5,
                    help="over the limit but within this factor is MARGINAL, beyond it FAILs")
    return ap


def main() -> int:
    a = build_args().parse_args()
    if a.streams < 1 or a.frame_ms < 1 or a.pace <= 0:
        raise SystemExit("--streams and --frame-ms must be >= 1 and --pace > 0")
    try:
        b0, b1 = (float(x) for x in a.class_bounds.split(","))
    except ValueError:
        raise SystemExit("--class-bounds wants two seconds, e.g. 8,20")
    classes = [c.strip() for c in a.bank.split(",") if c.strip()]
    unknown = [c for c in classes if c not in ("short", "medium", "long")]
    if unknown:
        raise SystemExit(f"unknown bank class(es): {','.join(unknown)}")

    clips = sorted(set(a.clips))
    if a.mode == "soak":
        bank = build_bank(clips, classes, (b0, b1))
        n_sched = max(len(clips) * 4, a.streams * 8)
        schedule = stratified_schedule(bank, classes, a.seed, n_sched)
    else:
        # WAVE keeps the v0 behaviour exactly: the clip list, cycled over streams and repeats
        bank = {classify(wav_audio_s(c), (b0, b1)): [] for c in clips}
        for c in clips:
            bank[classify(wav_audio_s(c), (b0, b1))].append((c, wav_audio_s(c)))
        schedule = clips
    classes_of = {c: classify(d, (b0, b1)) for v in bank.values() for c, d in v}

    chunk_ms = a.chunk_ms if a.chunk_ms else M.chunk_period_ms(int(a.lookahead))
    thr = M.default_thresholds(chunk_ms)
    if a.ttfp_p95_ms is not None:
        thr["ttfp_p95_ms"] = a.ttfp_p95_ms
    if a.lag_p95_ms is not None:
        thr["emission_lag_p95_ms"] = a.lag_p95_ms
    if a.backlog_max_s is not None:
        thr["backlog_max_s"] = a.backlog_max_s
    thr["finalization_p95_ms"] = a.fin_p95_ms
    thr["max_drift_pct"] = a.max_drift_pct
    thr["marginal_factor"] = a.marginal_factor

    health_before = health(a.host, a.port)
    a._t0 = time.monotonic()
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=stream_main, args=(a, i, schedule, classes_of, q), daemon=True)
             for i in range(a.streams)]
    for p in procs:
        p.start()

    records = []
    expected = a.streams * a.repeat if a.mode == "wave" else None
    # a blocking get with a timeout: the parent sleeps in the kernel, it never spins
    hard_deadline = a._t0 + (a.duration if a.mode == "soak" else a.repeat * a.done_timeout) \
        + a.done_timeout + 300.0
    while time.monotonic() < hard_deadline:
        if expected is not None and len(records) >= expected:
            break
        try:
            records.append(q.get(timeout=1.0))
        except Exception:                           # noqa: BLE001 — queue.Empty
            if not any(p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(10)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(5)
    while True:                                     # drain what landed during the join
        try:
            records.append(q.get_nowait())
        except Exception:                           # noqa: BLE001
            break
    wall = time.monotonic() - a._t0
    health_after = health(a.host, a.port)

    utts = [M.analyze_utterance(r, frame_ms=a.frame_ms, pace=a.pace, onsets=onsets)
            for r in records]
    warmup = a.warmup if a.mode == "soak" else 0.0
    window = a.window if a.mode == "soak" else None
    reference = json.load(open(a.reference)) if a.reference else None
    transcripts = load_transcripts(a.transcripts) if a.transcripts else None
    onsets = None
    if a.onsets:
        with open(a.onsets) as f:
            onsets = {k: float(v) for k, v in json.load(f).items()
                      if isinstance(v, (int, float))}
    summary = M.aggregate(utts, frame_ms=a.frame_ms, pace=a.pace, window_s=window,
                          warmup_s=warmup, t0=a._t0, reference=reference,
                          transcripts=transcripts)
    if expected is not None and len(records) < expected:
        summary["identity_fail"] = dict(summary["identity_fail"])
        summary["counts"]["missing"] = expected - len(records)
    env = M.envelope_verdict(summary, thr)
    if expected is not None and len(records) < expected:
        env["invalid_reasons"].append(f"{expected - len(records)} utterance(s) never reported")
        env["verdict"] = "INVALID"

    manifest = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "client_host": platform.node(), "client_platform": platform.platform(),
        "python": platform.python_version(), "tree": git_rev(),
        "server_url": f"http://{a.host}:{a.port}", "mode": a.mode.upper(),
        "concurrency": a.streams, "repeat": a.repeat if a.mode == "wave" else None,
        "duration_s": a.duration if a.mode == "soak" else None,
        "warmup_s": warmup, "window_s": window,
        "bank": {k: [{"clip": c, "audio_s": round(d, 3), "sha256": sha256(c)} for c, d in v]
                 for k, v in bank.items() if v},
        "bank_classes": classes if a.mode == "soak" else None,
        "class_bounds_s": [b0, b1], "seed": a.seed if a.mode == "soak" else None,
        "frame_ms": a.frame_ms, "pace": a.pace, "lang": a.lang, "lookahead": a.lookahead,
        "chunk_period_ms": chunk_ms, "thresholds": thr,
        "health_before": health_before, "health_after": health_after,
        "wall_s": wall,
    }

    tag = "WAVE (screening: may disqualify, never promotes)" if a.mode == "wave" \
        else f"SOAK (closed loop, {a.duration:.0f} s, the only gate that promotes)"
    print(f"stream_load {tag}")
    print(f"  server {manifest['server_url']}  concurrency={a.streams}  frame={a.frame_ms} ms  "
          f"pace={a.pace}  lookahead={a.lookahead}  chunk={chunk_ms:.0f} ms  "
          f"tree={manifest['tree']['head'][:12]}{'-dirty' if manifest['tree']['dirty'] else ''}")
    if a.mode == "soak":
        print(f"  bank seed={a.seed}  " + "  ".join(
            f"{k}:{len(v)}" for k, v in manifest["bank"].items()))
    print(f"  wall {wall:.1f} s")
    print(M.format_summary(summary, env, indent="  "))
    for name, d in sorted(summary.get("drift", {}).items()):
        if name != "emission_lag_ms" or not d.get("windows"):
            continue
        print("  per-window emission lag (client), p50/p95 ms:")
        for w in d["windows"]:
            p50 = "n/a" if w["p50"] is None else f"{w['p50']:.0f}"
            p95 = "n/a" if w["p95"] is None else f"{w['p95']:.0f}"
            print(f"    [{w['t0_s']:6.0f}-{w['t1_s']:6.0f} s] n={w['n']:<5} {p50} / {p95}")
    bad = [r for r in records if r.get("error") and not r.get("rejected")]
    for r in bad[:5]:
        print(f"  error stream {r.get('stream')} rep {r.get('rep')}: {r['error']}")
    if summary["counts"]["rejected"]:
        print(f"  rejections (503 before the upgrade, counted, not errors): "
              f"{summary['counts']['rejected']}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"manifest": manifest, "summary": summary, "envelope": env,
                       "utterances": utts}, f, indent=1)
        print(f"  -> {a.json}")

    return {"INVALID": 2, "NOT STREAMABLE": 1, "DEGRADED": 1}.get(env["verdict"], 0)


if __name__ == "__main__":
    sys.exit(main())
