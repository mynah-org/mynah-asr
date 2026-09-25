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
import math
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

# ------------------------------------------------------------------------ the clock
#
# CLOCK_MONOTONIC explicitly, never `time.monotonic()`, because the server stamps
# every arrival and every emission with `clock_gettime(CLOCK_MONOTONIC)`
# (`mynah_asr_now`, server/slot.c) and a waterfall that crosses the process
# boundary has to subtract two readings of ONE clock.
#
# On Linux the two are the same call and this changes nothing. On macOS
# `time.monotonic()` is `mach_absolute_time()`, which excludes the time the
# machine spent asleep, while `CLOCK_MONOTONIC` includes it: measured on this
# development host the two differ by 4.6 days. Durations inside one process are
# unaffected either way; a client mark minus a server mark is not.
def mono() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC)


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
    elif isinstance(doc, dict) and isinstance(doc.get("clips"), list):
        # samples/stress-en/manifest.json. Without this it fell to the generic
        # dict branch below and registered "source", "url" and "licence" as clip
        # names with their values as reference text.
        pairs = [(e.get("file"), e.get("text")) for e in doc["clips"]]
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


def schedule_stride(streams: int, period: int) -> int:
    """The step between two utterances of ONE stream in the shared schedule.

    Stream i plays positions i, i+s, i+2s, ... (mod the schedule length), and the
    stratified schedule owns class `p mod period` at position p. Until 2026-09-25 the step
    was s = streams, so when `streams` shared a factor with `period` every stream stayed
    in a subset of the classes for the whole run -- with three classes and C a multiple
    of 3 (96, 120, 144) each stream played ONE length class (AUDIT 2026-09-24, gap 4:
    at C=144 stream 62 played 81 long utterances, stream 99 261 short ones).

    Now s is the smallest step >= streams that is coprime with the period, so every
    stream walks all the classes in turn (i + k*s mod period takes every value) while the
    fleet still covers consecutive positions. Where `streams` was already coprime nothing
    changes: s == streams and the run replays exactly the audio it played before, so the
    C=128 and C=16 soaks stay reproducible; a C that is a multiple of 3 now gets
    s = C+1. Deterministic in (streams, period): the manifest records it."""
    s = max(1, int(streams))
    p = max(1, int(period))
    while math.gcd(s, p) != 1:
        s += 1
    return s


def schedule_clip(schedule, idx: int, k: int, stride: int):
    """The clip stream `idx` plays as its k-th utterance."""
    return schedule[(idx + k * stride) % len(schedule)]

# ---------------------------------------------------------------- one stream (one process)


def _fail(rec: dict, kind: str, msg: str) -> None:
    """Record the FIRST failure of an utterance and its kind (streaming_metrics.ERROR_KINDS).

    The first one wins: a send that fails because the server already closed is the
    consequence, and the close the reader saw is the cause."""
    if not rec["error"]:
        rec["error"] = msg
        rec["error_kind"] = kind


def _kind_of_oserror(e: BaseException) -> str:
    if isinstance(e, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(e, (ConnectionError, BrokenPipeError)):
        return "server_disconnect"      # EOF, reset, aborted, broken pipe
    return "client_exception"


def run_utterance(a, clip: str, pcm: bytes, cls: str) -> dict:
    """One utterance, recorded in the input format of `streaming_metrics`.

    Nothing is aggregated here: the record is raw marks.  `mono()` is CLOCK_MONOTONIC, which is
    system-wide, so marks taken in different stream processes share one timeline and can be
    windowed together by the parent -- and share it with the server's own stamps.

    Every way this can end without a `done` frame sets `error` AND `error_kind`. Until
    2026-09-25 a close frame before `done` made the reader stop without recording
    anything, and the utterance counted as OK (AUDIT 2026-09-24, gap 1)."""
    frame_bytes = int(16000 * a.frame_ms / 1000) * 2
    n_frames = max(1, (len(pcm) + frame_bytes - 1) // frame_bytes)
    rec = {"clip": clip, "class": cls, "audio_s": len(pcm) / 32000.0, "sends": [], "late_ms": [],
           "events": [], "done_t": None, "t_start": mono(), "error": None, "error_kind": None,
           "rejected": False, "status": None, "lang": None, "retry_after": None}
    # `model=` names a WORKER GROUP in a multi-model fleet (S2-6). Sent only when
    # asked for, so a single-model server sees exactly the query it always saw.
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
    if a.model:
        path += f"&model={a.model}"
    try:
        sock, status, headers = ws_connect(a.host, a.port, path, timeout=a.connect_timeout)
    except OSError as e:
        _fail(rec, "connect_error", f"connect: {e}")
        return rec
    rec["status"] = status
    if sock is None:
        # a refusal before the upgrade is the admission ladder working, not a failure
        rec["rejected"] = " 503 " in status
        rec["retry_after"] = headers.get("retry-after")
        _fail(rec, "rejected" if rec["rejected"] else "http_error", f"refused: {status}")
        return rec

    stop = threading.Event()

    def reader() -> None:
        try:
            while not stop.is_set():
                op, payload = ws_recv(sock)
                now = mono()
                if op == 0x8:
                    # the server closed before `done`: a lost utterance, never an OK one
                    code = struct.unpack(">H", payload[:2])[0] if len(payload) >= 2 else None
                    _fail(rec, "server_disconnect",
                          f"server close frame without done (code {code})")
                    break
                if op != 0x1:
                    continue
                try:
                    msg = json.loads(payload)
                except ValueError as e:
                    _fail(rec, "protocol_error", f"reader: unparsable text frame: {e}")
                    break
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
                    _fail(rec, "server_error",
                          f"server error: {msg.get('code')} {msg.get('message')}")
                    break
        except OSError as e:                         # timeout, EOF, reset
            if not stop.is_set():
                _fail(rec, _kind_of_oserror(e), f"reader: {e}")
        except Exception as e:                       # noqa: BLE001 — recorded, not raised
            if not stop.is_set():
                _fail(rec, "client_exception", f"reader: {type(e).__name__}: {e}")

    sock.settimeout(a.done_timeout)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    period = a.frame_ms / 1000.0 / a.pace
    t0 = mono()
    try:
        for i in range(n_frames):
            due = t0 + i * period
            now = mono()
            if due > now:
                time.sleep(due - now)              # the schedule, never a spin
            ws_send(sock, 0x2, pcm[i * frame_bytes:(i + 1) * frame_bytes])
            t_sent = mono()
            rec["sends"].append([t_sent, min((i + 1) * frame_bytes, len(pcm)) / 32000.0])
            rec["late_ms"].append(max(0.0, (t_sent - due) * 1000.0))
        ws_send(sock, 0x8, b"")                    # close = finalize, then `done`
        th.join(a.done_timeout)
        if th.is_alive():
            _fail(rec, "timeout", "timeout waiting for done")
    except OSError as e:
        th.join(0.5)                               # let the reader name the cause first
        _fail(rec, _kind_of_oserror(e), f"send: {e}")
    finally:
        stop.set()
        try:
            sock.close()
        except OSError:
            pass
    if rec["done_t"] is None:
        # the net under every path above: no `done` is never OK
        if rec["events"]:
            _fail(rec, "no_done", "the utterance ended without a done frame")
        else:
            _fail(rec, "no_events", "the utterance ended without any server frame")
    return rec


def stream_main(a, idx: int, schedule, classes_of, q, stride=None) -> None:
    """One stream process: WAVE plays `--repeat` utterances, SOAK loops until the deadline.

    The schedule is the parent's; this process only walks its own stride of it, so the audio
    a given stream played is reproducible from (seed, bank, streams, index).

    Accounting (AUDIT 2026-09-24, gaps 3 and 5): before each utterance the process puts a
    START mark on the queue, and when it leaves its loop an END mark. The parent reconciles
    the marks with the records (`streaming_metrics.reconcile`): a start with no record, or a
    stream with no end mark, cannot disappear from the counts any more -- in SOAK nothing
    used to check that a stream process lived to the deadline. A Python exception inside
    one utterance becomes a `client_exception` record and the stream goes on: a bug in the
    harness must be counted, not turned into a silently dead stream."""
    stride = a.streams if stride is None else stride
    pcms = {}
    deadline = a._t0 + a.duration if a.mode == "soak" else None
    k = 0
    reason = "repeat"
    while True:
        if a.mode == "wave" and k >= a.repeat:
            break
        if deadline is not None and mono() >= deadline:
            reason = "deadline"
            break
        clip = schedule_clip(schedule, idx, k, stride)
        cls = classes_of.get(clip, "n/a")
        q.put({"_mark": "start", "stream": idx, "rep": k, "t": mono(), "clip": clip,
               "class": cls})
        try:
            if clip not in pcms:
                pcms[clip] = load_pcm(clip)
            rec = run_utterance(a, clip, pcms[clip], cls)
        except Exception as e:                       # noqa: BLE001 — counted, not raised
            rec = {"clip": clip, "class": cls, "audio_s": 0.0, "sends": [], "late_ms": [],
                   "events": [], "done_t": None, "t_start": mono(), "rejected": False,
                   "error": f"client: {type(e).__name__}: {e}",
                   "error_kind": "client_exception"}
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
    q.put({"_mark": "end", "stream": idx, "t": mono(), "started": k, "reason": reason})

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

# ------------------------------------------------------------------------------ self-test
#
# `python3 tools/bench/stream_load.py --self-test` (part of `make check`): no model, no
# server binary. The schedule is checked for the stride lock, and the client is run
# against a fake WebSocket server on localhost that fails in each way the taxonomy names;
# every failure has to come back as an error of the right kind, never as OK.


def _fake_ws_server():
    """(port, closer). The scenario is the `lang=` of the query string."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    port = srv.getsockname()[1]

    def frame(op: int, payload: bytes) -> bytes:
        return bytes([0x80 | op, len(payload)]) + payload     # payloads here are < 126

    def read_frame(c):
        def rd(n):
            buf = b""
            while len(buf) < n:
                part = c.recv(n - len(buf))
                if not part:
                    raise ConnectionError("eof")
                buf += part
            return buf
        h = rd(2)
        n = h[1] & 0x7F
        if n == 126:
            n = struct.unpack(">H", rd(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", rd(8))[0]
        rd(4 + n)                                  # mask + payload, not needed
        return h[0] & 0x0F

    def serve(c):
        try:
            req = b""
            while b"\r\n\r\n" not in req:
                part = c.recv(4096)
                if not part:
                    return
                req += part
            line = req.split(b"\r\n", 1)[0].decode()
            if "lang=" not in line:                # e.g. the /v1/health probe
                c.sendall(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                return
            scen = line.split("lang=", 1)[1].split("&", 1)[0].split(" ", 1)[0]
            if scen in ("503", "500"):
                c.sendall((f"HTTP/1.1 {scen} X\r\nRetry-After: 0\r\n"
                           f"Content-Length: 0\r\n\r\n").encode())
                return
            c.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                      b"Connection: Upgrade\r\nSec-WebSocket-Accept: x\r\n\r\n")
            if scen == "close_early":              # close frame after the first audio frame
                read_frame(c)
                c.sendall(frame(0x1, b'{"type":"delta","text":"hi","audio_s":0.1}'))
                c.sendall(frame(0x8, struct.pack(">H", 1011)))
                time.sleep(0.2)
                return
            while read_frame(c) != 0x8:            # the whole utterance, up to the close
                pass
            if scen == "done":
                c.sendall(frame(0x1, b'{"type":"delta","text":"hi","audio_s":0.1}'))
                c.sendall(frame(0x1, b'{"type":"done","text":"hi"}'))
            elif scen == "close":                  # a close frame where `done` belongs
                c.sendall(frame(0x1, b'{"type":"delta","text":"hi","audio_s":0.1}'))
                c.sendall(frame(0x8, struct.pack(">H", 1000)))
            elif scen == "close_silent":           # a close frame and nothing else
                c.sendall(frame(0x8, struct.pack(">H", 1000)))
            elif scen == "eof":                    # TCP closed, no close frame
                pass
            elif scen == "error":
                c.sendall(frame(0x1, b'{"type":"error","code":500,"message":"boom"}'))
            elif scen == "garbage":
                c.sendall(frame(0x1, b"this is not json"))
            elif scen == "hang":                   # never answers
                time.sleep(1.5)
            time.sleep(0.1)
        except OSError:
            pass
        finally:
            try:
                c.close()
            except OSError:
                pass

    def accept():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=serve, args=(c,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return port, srv.close


def self_test() -> int:
    bad = 0

    def check(cond, what):
        nonlocal bad
        print(f"  {'ok  ' if cond else 'FAIL'} {what}")
        bad += 0 if cond else 1

    print("schedule: the stride cannot lock a stream to one length class (gap 4)")
    classes = ["short", "medium", "long"]
    bank = {k: [(f"{k}{j}.wav", 1.0) for j in range(7)] for k in classes}
    cls_of = {c: k for k, v in bank.items() for c, _ in v}
    for C in (3, 6, 16, 24, 64, 96, 120, 128, 144):
        sched = stratified_schedule(bank, classes, 42, max(21 * 4, C * 8))
        s = schedule_stride(C, len(classes))
        per = [[cls_of[schedule_clip(sched, i, k, s)] for k in range(30)] for i in range(C)]
        worst = min(min(p.count(k) for k in classes) for p in per)
        old = [[cls_of[schedule_clip(sched, i, k, C)] for k in range(30)] for i in range(C)]
        locked_old = sum(1 for p in old if len(set(p)) == 1)
        check(worst >= 8, f"C={C:<3} stride {s:<3}: every stream plays every class, "
                          f"rarest class {worst}/30 (old stride locked {locked_old}/{C})")
        if C % 3 == 0:
            check(locked_old == C, f"C={C:<3} planted: the OLD stride locks all {C} streams")
        else:
            check(s == C, f"C={C:<3} coprime with 3: stride unchanged, the run replays as before")
        # the fleet still holds one third per class at every round
        mix = [cls_of[schedule_clip(sched, i, 5, s)] for i in range(C)]
        check(max(mix.count(k) for k in classes) - min(mix.count(k) for k in classes) <= 1,
              f"C={C:<3} the fleet mix at a round stays balanced")
    check(schedule_stride(144, 3) == 145 and schedule_stride(128, 3) == 128
          and schedule_stride(7, 1) == 7, "stride values: 144 -> 145, 128 -> 128, WAVE unchanged")

    print("client: every way an utterance ends without `done` is an error of a named kind")
    port, close = _fake_ws_server()

    class A:
        host, lookahead, model = "127.0.0.1", "3", None
        connect_timeout, done_timeout, frame_ms, pace = 2.0, 0.7, 100, 50.0
    A.port = port
    pcm = b"\x00\x00" * 4800                        # 0.3 s: three frames
    for scen, want in (("done", None), ("close", "server_disconnect"),
                       ("close_silent", "server_disconnect"),
                       ("close_early", "server_disconnect"), ("eof", "server_disconnect"),
                       ("error", "server_error"), ("garbage", "protocol_error"),
                       ("hang", "timeout"), ("503", "rejected"), ("500", "http_error")):
        A.lang = scen
        rec = run_utterance(A, "x.wav", pcm, "short")
        u = M.analyze_utterance(rec, frame_ms=100, pace=1.0)
        got = M.outcome(u)
        check(got == (want or "ok") and (want is None) == (rec["error"] is None),
              f"{scen:<13} -> {got:<17} error={rec['error']!r}")
        if scen in ("close", "close_silent", "close_early"):
            # the planted fault of AUDIT gap 1: before 2026-09-25 the reader broke on the
            # close frame without recording anything, and `aggregate` counted these OK
            # (verified by running the old client against this server; EOF, by contrast,
            # already raised and was already an error)
            check(rec["done_t"] is None and bool(rec["error"]),
                  f"{scen:<13} has no done and now carries an error (it used to count OK)")
    close()
    A.port = port                                  # nobody listens there any more
    A.lang = "done"
    rec = run_utterance(A, "x.wav", pcm, "short")
    check(M.outcome(rec) == "connect_error", f"closed port  -> {M.outcome(rec)}")

    print(f"\nstream_load self-test: {'FAIL' if bad else 'PASS'} ({bad} failures)")
    return 1 if bad else 0

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
    ap.add_argument("--keep-events", type=int, default=0, metavar="N",
                    help="keep the published partials (type/t/text/audio_s/lag_ms) for "
                         "1 utterance in N, chosen by clip hash so the slice is the same "
                         "across runs, plus EVERY errored or rejected utterance. 0 = off")
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
    if "--self-test" in sys.argv[1:]:
        return self_test()
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
        stride = schedule_stride(a.streams, len(classes))
    else:
        # WAVE keeps the v0 behaviour exactly: the clip list, cycled over streams and repeats
        bank = {classify(wav_audio_s(c), (b0, b1)): [] for c in clips}
        for c in clips:
            bank[classify(wav_audio_s(c), (b0, b1))].append((c, wav_audio_s(c)))
        schedule = clips
        stride = a.streams
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
    a._t0 = mono()
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=stream_main, args=(a, i, schedule, classes_of, q, stride),
                        daemon=True)
             for i in range(a.streams)]
    for p in procs:
        p.start()

    records, starts, ends = [], [], {}

    def take(msg) -> None:
        mark = msg.get("_mark") if isinstance(msg, dict) else None
        if mark == "start":
            starts.append(msg)
        elif mark == "end":
            ends[msg["stream"]] = msg
        else:
            records.append(msg)

    expected = a.streams * a.repeat if a.mode == "wave" else None
    # a blocking get with a timeout: the parent sleeps in the kernel, it never spins
    hard_deadline = a._t0 + (a.duration if a.mode == "soak" else a.repeat * a.done_timeout) \
        + a.done_timeout + 300.0
    while mono() < hard_deadline:
        if expected is not None and len(records) >= expected:
            break
        try:
            take(q.get(timeout=1.0))
        except Exception:                           # noqa: BLE001 — queue.Empty
            if not any(p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(10)
    terminated = set()
    for i, p in enumerate(procs):
        if p.is_alive():
            # still running when collection closed: its utterance in flight is
            # `cut_at_deadline` in SOAK, never silently dropped
            terminated.add(i)
            p.terminate()
            p.join(5)
    while True:                                     # drain what landed during the join
        try:
            take(q.get_nowait())
        except Exception:                           # noqa: BLE001
            break
    wall = mono() - a._t0

    # Reconcile the marks with the records: every started utterance ends in exactly one
    # outcome, and every stream is accounted for (streaming_metrics.reconcile).
    status = {i: {"end_t": (ends.get(i) or {}).get("t"),
                  "started": (ends.get(i) or {}).get("started"),
                  "exitcode": p.exitcode, "terminated": i in terminated}
              for i, p in enumerate(procs)}
    longest = max((d for v in bank.values() for _, d in v), default=0.0)
    # "one utterance" of tolerance: the longest clip at this pace, plus the longest
    # Retry-After sleep a rejected stream takes before it looks at the clock again
    tol_s = longest / a.pace + 5.0
    n_reported = len(records)
    synth, streams_report = M.reconcile(
        records, starts, status, n_streams=a.streams,
        deadline=(a._t0 + a.duration) if a.mode == "soak" else None,
        tol_s=tol_s, repeat=a.repeat if a.mode == "wave" else None)
    records.extend(synth)
    health_after = health(a.host, a.port)

    # The onset map is READ FIRST: analyze_utterance takes it, and until R-2 ran
    # the harness for real it was loaded after the call that needs it, which made
    # every --onsets run die with a NameError instead of producing a number.
    onsets = None
    if a.onsets:
        with open(a.onsets) as f:
            onsets = {k: float(v) for k, v in json.load(f).items()
                      if isinstance(v, (int, float))}
    utts = [M.analyze_utterance(r, frame_ms=a.frame_ms, pace=a.pace, onsets=onsets)
            for r in records]
    # The published partials, kept for a DETERMINISTIC slice of the run and for
    # every utterance that went wrong. analyze_utterance folds the events into a
    # concatenated text and drops the sequence, which is exactly what a person
    # auditing a PASS later needs to see: what the server said, when, and how it
    # changed. Keeping them for everything would multiply a 15 MB soak record by
    # an order of magnitude, so the slice is by clip hash -- stable across runs,
    # so the same clips are auditable in the loaded run and in the reference.
    if a.keep_events:
        for u, r in zip(utts, records):
            clip = r.get("clip") or ""
            keep = bool(r.get("error") or r.get("rejected"))
            if not keep and a.keep_events > 0:
                h = int(hashlib.sha1(clip.encode()).hexdigest()[:8], 16)
                keep = (h % a.keep_events) == 0
            if keep:
                u["events"] = [{k: e.get(k) for k in ("type", "t", "text", "audio_s", "lag_ms")}
                               for e in (r.get("events") or [])]
                u["t_start_abs"] = r.get("t_start")
    warmup = a.warmup if a.mode == "soak" else 0.0
    window = a.window if a.mode == "soak" else None
    reference = json.load(open(a.reference)) if a.reference else None
    transcripts = load_transcripts(a.transcripts) if a.transcripts else None
    summary = M.aggregate(utts, frame_ms=a.frame_ms, pace=a.pace, window_s=window,
                          warmup_s=warmup, t0=a._t0, reference=reference,
                          transcripts=transcripts, started=len(starts),
                          streams=streams_report)
    if expected is not None and n_reported < expected:
        summary["counts"]["missing"] = expected - n_reported
    env = M.envelope_verdict(summary, thr)
    if expected is not None and n_reported < expected:
        env["invalid_reasons"].append(f"{expected - n_reported} utterance(s) never reported")
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
        # stream i plays schedule positions i + k*stride (schedule_stride); recorded so
        # a run says which of the two schedules it played
        "schedule_stride": stride, "schedule_len": len(schedule),
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
    bad = [u for u in utts if M.is_error(M.outcome(u))]
    for u in bad[:5]:
        print(f"  error stream {u.get('stream')} rep {u.get('rep')} "
              f"[{u.get('error_kind')}]: {u['error']}")
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
