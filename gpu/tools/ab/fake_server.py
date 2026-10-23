#!/usr/bin/env python3
"""fake_server.py -- a CPU stand-in for mynah-asr-server-cuda, for the harness self-test.

    gpu/tools/ab/fake_server.py -m PACK -p PORT [any server flag ...]
        [--fake-delay-ms N] [--fake-salt S] [--fake-die-after N]

It exists so gpu/tools/ab/test_ab_harness.sh can drive ab_server.sh end to end
(start, health, load, stop, meta, report) on a laptop before a GPU minute is
spent. It is NOT a model: a delta's text is a hash of the audio it closes, so
the same clip always gives the same transcript (and --fake-salt changes it),
--fake-delay-ms holds every delta back (a slower arm), --fake-die-after N
exits after the N-th session starts (a crash). /v1/health carries the
counters the report reads (steps, batch, engine.device_errors_total).
Unknown flags are accepted and ignored, as the real server's are not.
Standard library only."""
import hashlib
import json
import socket
import struct
import sys
import threading
import time

args = sys.argv[1:]


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


PORT = int(opt("-p", "8391"))
DELAY = float(opt("--fake-delay-ms", "0")) / 1000.0
SALT = opt("--fake-salt", "").encode()
DIE_AFTER = int(opt("--fake-die-after", "0"))
CHUNK = 16000 * 2 // 2            # a delta per 0.5 s of s16 audio

lock = threading.Lock()
books = {"sessions": 0, "completed": 0, "active": 0, "steps": 0, "deltas": 0, "audio": 0.0}


def frame(op, payload):
    n = len(payload)
    if n < 126:
        return bytes([0x80 | op, n]) + payload
    return bytes([0x80 | op, 126]) + struct.pack(">H", n) + payload


def rd(c, n):
    buf = b""
    while len(buf) < n:
        part = c.recv(n - len(buf))
        if not part:
            raise ConnectionError("eof")
        buf += part
    return buf


def read_frame(c):
    h = rd(c, 2)
    n = h[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", rd(c, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", rd(c, 8))[0]
    mask = rd(c, 4) if h[1] & 0x80 else b"\0\0\0\0"
    data = rd(c, n)
    return h[0] & 0x0F, bytes(b ^ mask[i % 4] for i, b in enumerate(data))


def health():
    with lock:
        return {"status": "ok", "worker": -1, "inflight": books["active"],
                "slots": {"active": books["active"], "cap": 64},
                "steps": books["steps"], "deltas": books["deltas"], "eous": 0,
                "sessions": books["sessions"], "completed": books["completed"],
                "aborted": 0, "cancelled": 0,
                "balanced": books["sessions"] == books["completed"] + books["active"],
                "cancelled_by": {}, "audio_seconds": books["audio"],
                "batch": {"batched_steps_total": books["steps"], "ready_sum": books["steps"],
                          "step_wall_ms": {"sum": books["steps"] * 2.0, "count": books["steps"]}},
                "engine": {"name": "fake", "device_errors_total": 0}}


def delta(c, pcm, consumed):
    if DELAY:
        time.sleep(DELAY)
    word = " w" + hashlib.sha1(SALT + pcm).hexdigest()[:6]
    with lock:
        books["steps"] += 1
        books["deltas"] += 1
    c.sendall(frame(0x1, json.dumps({"type": "delta", "text": word, "audio_s": consumed,
                                     "lag_ms": DELAY * 1000.0}).encode()))


def serve(c):
    try:
        req = b""
        while b"\r\n\r\n" not in req:
            part = c.recv(4096)
            if not part:
                return
            req += part
        line = req.split(b"\r\n", 1)[0].decode()
        if line.startswith("GET /v1/health"):
            body = json.dumps(health()).encode()
            c.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                      + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
            return
        c.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                  b"Connection: Upgrade\r\nSec-WebSocket-Accept: x\r\n\r\n")
        with lock:
            books["sessions"] += 1
            books["active"] += 1
            n_sess = books["sessions"]
        if DIE_AFTER and n_sess >= DIE_AFTER:
            print("fake_server: dying on purpose", file=sys.stderr, flush=True)
            import os
            os._exit(3)
        pending, total = b"", 0
        while True:
            op, payload = read_frame(c)
            if op == 0x2:
                pending += payload
                while len(pending) >= CHUNK:
                    total += CHUNK
                    delta(c, pending[:CHUNK], total / 32000.0)
                    pending = pending[CHUNK:]
            elif op == 0x8:
                if pending:
                    total += len(pending)
                    delta(c, pending, total / 32000.0)
                c.sendall(frame(0x1, json.dumps({"type": "done", "lang": "xx"}).encode()))
                with lock:
                    books["active"] -= 1
                    books["completed"] += 1
                    books["audio"] += total / 32000.0
                return
    except (OSError, ConnectionError):
        with lock:
            books["active"] = max(0, books["active"] - 1)
    finally:
        try:
            c.close()
        except OSError:
            pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(256)
    print(f"[SERVER-CONFIG] mynah-asr-server-cuda fake: engine=cuda device=\"fake\" precision=bf16 "
          f"gemm=own-tc args={' '.join(args)}", file=sys.stderr, flush=True)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=serve, args=(c,), daemon=True).start()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    main()
