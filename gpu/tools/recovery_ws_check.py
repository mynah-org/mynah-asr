#!/usr/bin/env python3
"""recovery_ws_check.py -- the GPU server's empty-turn recovery seen from a client (S18).

    # a bank of clips, several turns per session (finalize between turns), N sessions at once
    python3 gpu/tools/recovery_ws_check.py run --port 8291 --clips bank/*.wav --turns 4 \
        --sessions 8 --out off.json
    # the S18 items (id lang ctx_wav cur_wav start end): one session per item, turn 1 =
    # the ctx clip, turn 2 = the current turn; .f32 paths (test_cuda_empty_recovery
    # --dump-dir) are sent as f32le, the exact samples that test decoded
    python3 gpu/tools/recovery_ws_check.py run --port 8291 --items items.tsv --out on.json
    # byte identity of everything but timing: lag_ms, lag_p50_ms, lag_max_ms
    python3 gpu/tools/recovery_ws_check.py compare off.json on.json

Every frame the server sends (deltas, done, errors) is kept per session in order, minus
the timing fields; `compare` exits 0 when two runs are identical, 1 with the first
differences otherwise. `run` prints how many `done` frames say "recovered". Clips are sent
as fast as the server takes them (the server's ring throttles), so the run is about
transcripts, never latency. Standard library only; WAVs must be mono s16 (other rates are resampled linearly to 16 kHz).
"""
from __future__ import annotations

import argparse
import array
import base64
import json
import os
import socket
import struct
import sys
import threading
import wave

TIMING = {"lag_ms", "lag_p50_ms", "lag_max_ms"}


def ws_connect(host: str, port: int, path: str) -> socket.socket:
    s = socket.create_connection((host, port), timeout=60)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        c = s.recv(1)
        if not c:
            raise ConnectionError("closed during the handshake")
        head += c
    if b" 101 " not in head.split(b"\r\n", 1)[0]:
        raise ConnectionError(head.decode(errors="replace"))
    return s


def ws_send(s: socket.socket, op: int, payload: bytes) -> None:
    mask = os.urandom(4)
    n = len(payload)
    hdr = bytes([0x80 | op])
    if n < 126:
        hdr += bytes([0x80 | n])
    elif n < 65536:
        hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
    s.sendall(hdr + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(payload)))


def ws_recv(s: socket.socket):
    def rd(n: int) -> bytes:
        b = b""
        while len(b) < n:
            c = s.recv(n - len(b))
            if not c:
                raise ConnectionError("closed")
            b += c
        return b
    h = rd(2)
    op, n = h[0] & 0x0F, h[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", rd(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", rd(8))[0]
    return op, rd(n)


def load_pcm(path: str, t0: float = 0.0, t1: float = 0.0) -> bytes:
    if path.endswith(".f32"):   # raw 16 kHz f32le samples (test_cuda_empty_recovery --dump-dir)
        with open(path, "rb") as f:
            return f.read()
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need mono s16")
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    if sr != 16000:
        # linear interpolation to 16 kHz (a client-side resample; the server takes 16 kHz only)
        x = array.array("h", pcm)
        n = int(len(x) * 16000 / sr)
        y = array.array("h", bytes(2 * n))
        for i in range(n):
            pos = i * sr / 16000
            k = int(pos)
            f = pos - k
            a0 = x[k] if k < len(x) else 0
            a1 = x[k + 1] if k + 1 < len(x) else a0
            y[i] = int(round(a0 + (a1 - a0) * f))
        pcm = y.tobytes()
    a = int(t0 * 16000) * 2
    b = int(t1 * 16000) * 2 if t1 > 0 else len(pcm)
    return pcm[a:b]


def session(a, sid: int, lang: str, turns, out: dict) -> None:
    q = f"lang={lang}&lookahead={a.lookahead}" + (f"&{a.query}" if a.query else "") + \
        ("&format=f32le" if a.f32 else "")
    frames = []
    try:
        s = ws_connect(a.host, a.port, "/v1/audio/stream?" + q)
        for pcm in turns:
            step = 12800 if a.f32 else 6400
            for off in range(0, len(pcm), step):
                ws_send(s, 0x2, pcm[off:off + step])
            ws_send(s, 0x1, b'{"type":"finalize"}')
            while True:
                op, payload = ws_recv(s)
                if op != 0x1:
                    continue
                j = json.loads(payload)
                frames.append({k: v for k, v in j.items() if k not in TIMING})
                if j.get("type") in ("done", "error"):
                    break
        ws_send(s, 0x8, b"")
        try:
            while True:
                op, payload = ws_recv(s)
                if op == 0x8:
                    break
                if op == 0x1:
                    j = json.loads(payload)
                    frames.append({k: v for k, v in j.items() if k not in TIMING})
        except (ConnectionError, OSError):
            pass
        s.close()
    except Exception as e:  # noqa: BLE001 -- recorded, the compare shows it
        frames.append({"client_error": str(e)})
    out[sid] = frames


def cmd_run(a) -> int:
    plans = []   # (lang, [pcm per turn], label)
    if a.items:
        for line in open(a.items):
            f = line.rstrip("\n").split("\t")
            if len(f) < 6 or f[0].startswith("#"):
                continue
            plans.append((f[1], [load_pcm(f[2]), load_pcm(f[3], float(f[4]), float(f[5]))], f[0]))
            a.f32 = f[2].endswith(".f32")
    else:
        clips = sorted(a.clips)
        for i in range(0, len(clips), a.turns):
            group = clips[i:i + a.turns]
            plans.append((a.lang, [load_pcm(c) for c in group], ",".join(os.path.basename(c) for c in group)))
    out: dict = {}
    if not a.items:
        a.f32 = False
    for b in range(0, len(plans), a.sessions):
        th = [threading.Thread(target=session, args=(a, b + k, plans[b + k][0], plans[b + k][1], out))
              for k in range(min(a.sessions, len(plans) - b))]
        for t in th:
            t.start()
        for t in th:
            t.join()
    res = {plans[i][2]: out[i] for i in range(len(plans))}
    with open(a.out, "w") as fo:
        json.dump(res, fo, indent=1, sort_keys=True, ensure_ascii=False)
    dones = [fr for v in res.values() for fr in v if fr.get("type") == "done"]
    rec = [fr for fr in dones if fr.get("recovered")]
    errs = [fr for v in res.values() for fr in v if fr.get("type") == "error" or "client_error" in fr]
    kinds = {}
    for fr in rec:
        kinds[fr.get("recovery")] = kinds.get(fr.get("recovery"), 0) + 1
    print(f"sessions {len(res)} done frames {len(dones)} recovered {len(rec)} {kinds} errors {len(errs)} -> {a.out}")
    return 0 if not errs else 1


def cmd_compare(a) -> int:
    x, y = json.load(open(a.a)), json.load(open(a.b))
    diffs = 0
    for k in sorted(set(x) | set(y)):
        if x.get(k) != y.get(k):
            diffs += 1
            if diffs <= 5:
                print(f"DIFF {k}:\n  a: {json.dumps(x.get(k), ensure_ascii=False)[:600]}\n"
                      f"  b: {json.dumps(y.get(k), ensure_ascii=False)[:600]}")
    print(f"{len(set(x) | set(y))} sessions, {diffs} differ")
    return 0 if diffs == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8291)
    r.add_argument("--clips", nargs="*", default=[])
    r.add_argument("--items")
    r.add_argument("--turns", type=int, default=4, help="clips per session (--clips)")
    r.add_argument("--sessions", type=int, default=8, help="sessions at once")
    r.add_argument("--lang", default="auto")
    r.add_argument("--lookahead", default="3")
    r.add_argument("--query", default="", help="extra query, e.g. recover=0")
    r.add_argument("--out", required=True)
    r.add_argument("--f32", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == "run" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
