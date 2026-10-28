#!/usr/bin/env python3
"""Self-checking WebSocket probes for mynah-asr-server, protocol v2 (S2-5).

Python stdlib only, on purpose: `tests/test_server_protocol.sh` must run on a
machine that has a compiler and nothing else. Each subcommand drives ONE
behaviour of the protocol, asserts it itself and exits 0 or 1 with a line saying
what it saw — the shell test then only has to report the verdict, which keeps
the assertions next to the bytes that produced them.

  utterances   three utterances on one socket, separated by finalize/reset,
               each byte-identical to the reference, `seq` continuing
  control      an unknown control message is an error frame and the session
               survives it
  bad-lang     a reset naming a language this model does not hold is an error
               frame and the session survives it
  http         a raw HTTP request: the status, headers and body BEFORE any
               upgrade (unknown query key -> 400, cap reached -> 503)
  hold         take a slot and keep it (the other half of the 503 test), or
               stream a clip and wait for what the server says on the way out
               (SIGTERM -> `error shutting_down`)
  idle         say nothing after the handshake and be cancelled
  ping         read for a while and count the server's pings
  recovery     --empty-recovery (opt-in): starts its own two servers, flag off
               and flag on, and checks the protocol -- off is a plain `done`,
               noise never recovers, a healthy turn is byte-identical, an
               EMPTY-prone turn recovers or ends within the deadline

Usage: python3 tests/ws_probe.py <command> [options]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import random
import socket
import subprocess
import urllib.request
import struct
import sys
import threading
import time
import wave

FRAME_MS = 100


# --------------------------------------------------------------- the wire

def ws_frame(opcode: int, payload: bytes) -> bytes:
    mask = os.urandom(4)
    header = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", n)
    return header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def http_request(host: str, port: int, path: str, timeout: float = 15.0) -> tuple[str, dict, bytes]:
    """One GET with the WebSocket upgrade headers; returns (status line, headers, body).

    Used for the refusals that happen BEFORE the 101: the point of those is that
    the client reads an HTTP status and a readable body, so the probe reads them
    exactly as any HTTP client would and never as a WebSocket.
    """
    sock = socket.create_connection((host, port), timeout=timeout)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                  f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    try:
        while b"\r\n\r\n" not in buf:
            part = sock.recv(4096)
            if not part:
                break
            buf += part
        head, _, body = buf.partition(b"\r\n\r\n")
        lines = head.decode(errors="replace").split("\r\n")
        status = lines[0] if lines else ""
        headers = {}
        for line in lines[1:]:
            k, _, v = line.partition(":")
            if k:
                headers[k.strip().lower()] = v.strip()
        want = int(headers.get("content-length", "0") or 0)
        while len(body) < want:
            part = sock.recv(4096)
            if not part:
                break
            body += part
        return status, headers, body
    finally:
        sock.close()


class Session:
    """One WebSocket session, with a reader thread so a blasting sender can never
    deadlock against the server's own output."""

    def __init__(self, host: str, port: int, path: str, timeout: float = 60.0,
                 pong: bool = True):
        # pong=False is a client that reads and says NOTHING back, which is the
        # only way to observe --idle-ms while the server is pinging: the server
        # counts a pong as activity like any other frame, so a client that
        # answers its pings is by definition not idle.
        self.pong = pong
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                           f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
                          .encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            part = self.sock.recv(4096)
            if not part:
                break
            resp += part
        self.status = resp.split(b"\r\n", 1)[0].decode(errors="replace")
        if " 101 " not in self.status:
            self.sock.close()
            raise RuntimeError(f"handshake refused: {self.status}")

        self.msgs: queue.Queue = queue.Queue()
        self.pings = 0
        self.peer_closed = threading.Event()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # -- sending -----------------------------------------------------------
    def _send(self, opcode: int, payload: bytes) -> None:
        with self._send_lock:
            self.sock.sendall(ws_frame(opcode, payload))

    def send_binary(self, data: bytes) -> None:
        self._send(0x2, data)

    def send_text(self, obj) -> None:
        self._send(0x1, json.dumps(obj).encode())

    def send_close(self) -> None:
        try:
            self._send(0x8, b"")
        except OSError:
            pass

    # -- receiving ---------------------------------------------------------
    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            part = self.sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("closed")
            buf += part
        return buf

    def _read_loop(self) -> None:
        try:
            while not self._stop.is_set():
                h = self._recv_exact(2)
                opcode = h[0] & 0x0F
                plen = h[1] & 0x7F
                if plen == 126:
                    plen = struct.unpack(">H", self._recv_exact(2))[0]
                elif plen == 127:
                    plen = struct.unpack(">Q", self._recv_exact(8))[0]
                payload = self._recv_exact(plen) if plen else b""
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    self.pings += 1
                    if self.pong:
                        try:
                            self._send(0xA, payload)
                        except OSError:
                            break
                    continue
                if opcode == 0xA:
                    continue
                if opcode == 0x1:
                    try:
                        self.msgs.put(json.loads(payload))
                    except ValueError:
                        self.msgs.put({"type": "unparseable", "raw": payload.decode(
                            "utf-8", "replace")})
        except (OSError, ConnectionError):
            pass
        finally:
            self.peer_closed.set()

    def next_msg(self, timeout: float):
        try:
            return self.msgs.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


# ------------------------------------------------------------- utilities

def load_pcm(path: str) -> bytes:
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need a 16 kHz mono s16 WAV")
        return w.readframes(w.getnframes())


def send_clip(sess: Session, pcm: bytes, pace: float = 0.0) -> None:
    """Sends the whole clip as binary frames. pace=0 blasts it: the slot's ring is
    30 s deep, so a test clip never blocks, and the transcript does not depend on
    the pacing (the scheduler consumes fixed chunks). Pacing is for the cadence
    measurements in tools/bench/stream_load.py, not for identity."""
    step = int(16000 * FRAME_MS / 1000) * 2
    for off in range(0, len(pcm), step):
        sess.send_binary(pcm[off:off + step])
        if pace > 0:
            time.sleep(FRAME_MS / 1000.0 / pace)


def collect_utterance(sess: Session, timeout: float = 60.0):
    """Reads frames until `done`. Returns (text, seqs, error_codes)."""
    texts, seqs, errors = [], [], []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = sess.next_msg(timeout=max(0.1, deadline - time.monotonic()))
        if msg is None:
            break
        if "seq" in msg:
            seqs.append(msg["seq"])
        kind = msg.get("type")
        if kind == "error":
            errors.append(msg.get("code"))
            continue
        if kind == "done":
            return "".join(texts), seqs, errors
        if kind == "delta":
            texts.append(msg.get("text", ""))
    return "".join(texts), seqs, errors + ["timeout"]


def load_reference(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def stream_path(a) -> str:
    """The stream URL; an empty --lang or --lookahead leaves that key out, for a
    model with no language prompt or a single preset (the English-only EOU
    model refuses any `lang`)."""
    q = [f"{k}={v}" for k, v in (("lang", a.lang), ("lookahead", a.lookahead)) if v != ""]
    return "/v1/audio/stream" + ("?" + "&".join(q) if q else "")


def verdict(ok: bool, what: str) -> int:
    print(("OK   " if ok else "FAIL ") + what)
    return 0 if ok else 1


# ------------------------------------------------------------- commands

def cmd_utterances(a) -> int:
    ref = load_reference(a.reference)
    path = stream_path(a)
    sess = Session(a.host, a.port, path)
    bad, all_seqs = [], []
    try:
        for i, clip in enumerate(a.clips):
            if i > 0:
                sess.send_text({"type": "reset"})
            send_clip(sess, load_pcm(clip))
            sess.send_text({"type": "finalize"})
            text, seqs, errors = collect_utterance(sess, a.timeout)
            all_seqs += seqs
            if errors:
                bad.append(f"utterance {i + 1} ({clip}): error {errors}")
            expect = ref.get(clip)
            if expect is None:
                bad.append(f"utterance {i + 1}: no reference for {clip}")
            elif text != expect:
                bad.append(f"utterance {i + 1} ({clip}): {text!r} != {expect!r}")
        sess.send_close()
    finally:
        sess.close()
    if all_seqs != sorted(all_seqs) or len(set(all_seqs)) != len(all_seqs):
        bad.append(f"seq did not continue across the utterances: {all_seqs}")
    print(f"     seq {all_seqs[0] if all_seqs else '-'}..{all_seqs[-1] if all_seqs else '-'} "
          f"over {len(a.clips)} utterances on one socket")
    for b in bad:
        print("     " + b)
    return verdict(not bad, f"{len(a.clips)} utterances on one socket, identity and seq")


def cmd_control(a) -> int:
    ref = load_reference(a.reference)
    path = stream_path(a)
    sess = Session(a.host, a.port, path)
    bad = []
    try:
        sess.send_text({"type": a.message})
        msg = sess.next_msg(timeout=10.0)
        if msg is None or msg.get("type") != "error" or msg.get("code") != a.expect_code:
            bad.append(f"expected an error frame with code {a.expect_code}, got {msg}")
        # ...and the session must still work: one full utterance after it.
        send_clip(sess, load_pcm(a.clip))
        sess.send_text({"type": "finalize"})
        text, _, errors = collect_utterance(sess, a.timeout)
        if errors:
            bad.append(f"the session did not survive: {errors}")
        if text != ref.get(a.clip):
            bad.append(f"text after the refused message: {text!r} != {ref.get(a.clip)!r}")
        sess.send_close()
    finally:
        sess.close()
    for b in bad:
        print("     " + b)
    return verdict(not bad, f"{a.expect_code} is an error frame and the session survives it")


def cmd_bad_lang(a) -> int:
    ref = load_reference(a.reference)
    path = stream_path(a)
    sess = Session(a.host, a.port, path)
    bad = []
    try:
        sess.send_text({"type": "reset", "lang": a.bad})
        msg = sess.next_msg(timeout=10.0)
        if msg is None or msg.get("type") != "error" or msg.get("code") != "language_not_served":
            bad.append(f"expected language_not_served, got {msg}")
        send_clip(sess, load_pcm(a.clip))
        sess.send_text({"type": "finalize"})
        text, _, errors = collect_utterance(sess, a.timeout)
        if errors:
            bad.append(f"the session did not survive: {errors}")
        if text != ref.get(a.clip):
            bad.append(f"text after the refused reset: {text!r} != {ref.get(a.clip)!r}")
        sess.send_close()
    finally:
        sess.close()
    for b in bad:
        print("     " + b)
    return verdict(not bad, "a reset naming an unserved language is an error frame, not a hangup")


def cmd_http(a) -> int:
    status, headers, body = http_request(a.host, a.port, a.path)
    text = body.decode("utf-8", "replace")
    bad = []
    if f" {a.expect_status} " not in status + " ":
        bad.append(f"status {status!r}")
    for needle in a.expect_body or []:
        if needle not in text:
            bad.append(f"{needle!r} missing from the body")
    for h in a.expect_header or []:
        if h.lower() not in headers:
            bad.append(f"no {h} header")
    print(f"     {status.strip()} | {text.strip()[:220]}")
    for b in bad:
        print("     " + b)
    return verdict(not bad, f"{a.expect_status} before the upgrade")


def cmd_hold(a) -> int:
    """Takes a slot and keeps it. With --clip it streams one first, which is what
    the SIGTERM check needs: a live stream when the signal arrives."""
    path = stream_path(a)
    sess = Session(a.host, a.port, path)
    codes = []
    try:
        if a.ready_file:
            with open(a.ready_file, "w", encoding="utf-8") as f:
                f.write("ready\n")
        if a.clip:
            try:
                send_clip(sess, load_pcm(a.clip), pace=a.pace)
            except OSError:
                # The server closing under a sender is exactly the case this
                # probe exists for (SIGTERM mid-stream). What matters is what it
                # SAID on the way out, and the reader thread already has it: the
                # frames it wrote are in this socket's receive buffer whether or
                # not our next write succeeds.
                pass
        deadline = time.monotonic() + a.hold
        while time.monotonic() < deadline:
            msg = sess.next_msg(timeout=max(0.1, deadline - time.monotonic()))
            if msg is None:
                if sess.peer_closed.is_set():
                    break
                continue
            if msg.get("type") == "error":
                codes.append(msg.get("code"))
                if a.expect_code and msg.get("code") == a.expect_code:
                    break
    finally:
        sess.close()
    if not a.expect_code:
        return verdict(True, "slot held")
    print(f"     error codes seen: {codes}")
    return verdict(a.expect_code in codes,
                   f"a live stream is told `{a.expect_code}` on the way out")


def cmd_idle(a) -> int:
    """Says nothing at all after the handshake."""
    path = stream_path(a)
    sess = Session(a.host, a.port, path, pong=False)
    codes = []
    try:
        deadline = time.monotonic() + a.wait
        while time.monotonic() < deadline:
            msg = sess.next_msg(timeout=max(0.1, deadline - time.monotonic()))
            if msg is None:
                if sess.peer_closed.is_set():
                    break
                continue
            if msg.get("type") == "error":
                codes.append(msg.get("code"))
                break
    finally:
        sess.close()
    print(f"     error codes seen: {codes}, peer closed: {sess.peer_closed.is_set()}")
    return verdict("idle_timeout" in codes, "an idle client is cancelled with idle_timeout")


def cmd_ping(a) -> int:
    """Reads for a while without saying anything else and counts server pings."""
    path = stream_path(a)
    sess = Session(a.host, a.port, path)
    try:
        time.sleep(a.wait)
        seen = sess.pings
    finally:
        sess.close()
    print(f"     {seen} ping(s) in {a.wait} s")
    return verdict(seen >= a.min_pings,
                   f"the server pings (>= {a.min_pings} in {a.wait} s)")


# ------------------------------------------------------- empty-turn recovery

# Fields that carry wall-clock timing: the only ones allowed to differ between
# two runs of the same turn.
TIMING_KEYS = ("lag_ms", "lag_p50_ms", "lag_max_ms")


def has_content(text: str) -> bool:
    """The server's own notion of a non-empty turn: a letter or a digit."""
    return any(ch.isalnum() for ch in text)


def speech_onset(pcm: bytes, thr: float = 600.0) -> int:
    """Byte offset of the first 10 ms window above `thr` RMS, minus 50 ms."""
    n = len(pcm) // 2
    s = struct.unpack("<%dh" % n, pcm[:n * 2])
    for i in range(0, n - 160, 160):
        w = s[i:i + 160]
        if (sum(v * v for v in w) / 160.0) ** 0.5 > thr:
            return max(0, i - 800) * 2
    return 0


def noise_pcm(seconds: float, sigma: float, seed: int) -> bytes:
    rng = random.Random(seed)
    n = int(16000 * seconds)
    return struct.pack("<%dh" % n, *[max(-32768, min(32767, int(rng.gauss(0.0, sigma))))
                                     for _ in range(n)])


def turn_frames(sess: Session, timeout: float):
    """Reads one utterance's frames, through `done`. Returns (frames, t_done)."""
    frames = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        msg = sess.next_msg(timeout=max(0.1, deadline - time.monotonic()))
        if msg is None:
            break
        frames.append(msg)
        if msg.get("type") in ("done", "error"):
            return frames, time.monotonic()
    return frames, None


def run_turns(host, port, path, clips, timeout):
    """Sends each clip then a finalize on ONE socket without waiting in between
    (a second turn queues behind a held one), and reads every turn's frames.
    Returns [(frames, seconds from that turn's finalize to its done)]."""
    sess = Session(host, port, path)
    out = []
    try:
        t_fin = []
        for pcm in clips:
            send_clip(sess, pcm)
            sess.send_text({"type": "finalize"})
            t_fin.append(time.monotonic())
        for i in range(len(clips)):
            frames, t_done = turn_frames(sess, timeout)
            out.append((frames, None if t_done is None else t_done - t_fin[i]))
        sess.send_close()
    finally:
        sess.close()
    return out


def strip_timing(frames):
    return [{k: v for k, v in f.items() if k not in TIMING_KEYS} for f in frames]


def text_of(frames) -> str:
    return "".join(f.get("text", "") for f in frames if f.get("type") == "delta")


def health(host, port) -> dict:
    with urllib.request.urlopen(f"http://{host}:{port}/v1/health", timeout=10) as r:
        return json.loads(r.read())


class Server:
    """One mynah-asr-server for the duration of a `with`, stopped by its pid."""

    def __init__(self, a, port, extra):
        self.port = port
        cmd = [a.server_bin, "-m", a.model_dir, "-p", str(port), "--threads", "4",
               "--batch", "1", "--cap", "2", "--idle-ms", "30000", "--ping-ms", "0"]
        if a.quant:
            cmd += ["--quant", a.quant]
        self.log = open(os.path.join(a.log_dir, f"recovery_srv_{port}.log"), "w")
        self.p = subprocess.Popen(cmd + extra, stdout=self.log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL)
        t_end = time.monotonic() + 60
        while time.monotonic() < t_end:
            if self.p.poll() is not None:
                raise RuntimeError(f"server on {port} exited with {self.p.returncode}")
            try:
                health(a.host, port)
                return
            except OSError:
                time.sleep(0.2)
        self.stop()
        raise RuntimeError(f"server on {port} never became ready")

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait(timeout=5)
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def wait_held(a, port, want: int, timeout: float) -> bool:
    """Polls /v1/health until recovery.held == want."""
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        rv = health(a.host, port).get("recovery") or {}
        if rv.get("held") == want:
            return True
        time.sleep(0.02)
    return False


def check_books(rv, bad, where):
    if not isinstance(rv, dict):
        bad.append(f"{where}: /v1/health has no recovery block")
        return
    acc = rv["accepted"]["phase"] + rv["accepted"]["replay"]
    ends = acc + rv["rejected"] + rv["failed"] + rv["deadline"] + rv["cancelled"] + rv["held"]
    if rv["attempted"] != ends:
        bad.append(f"{where}: recovery books do not balance: {rv}")
    acc = rv["accepted"]["phase"] + rv["accepted"]["replay"]
    print(f"     {where} health recovery: {json.dumps(rv, sort_keys=True)}")
    print(f"     {where} lane: attempted={rv['attempted']} accepted={acc} "
          f"skipped_busy={rv['skipped_busy']} lane_jobs={rv['lane_jobs']} "
          f"seconds_total={rv['seconds_total']:.3f} "
          f"seconds_per_job={rv['seconds_per_job']:.3f}")


def check_empty_turn(frames, secs, limit, bad, where):
    """The protocol of an EMPTY-prone turn on a server with recovery on: either
    a final delta with text then `done` with recovered/recovery, or a plain
    `done` within `limit` seconds (None = no time bound)."""
    d = frames[-1] if frames else {}
    if d.get("type") != "done":
        bad.append(f"{where}: no done: {frames}")
    elif d.get("recovered") is True:
        finals = [f for f in frames if f.get("type") == "delta" and f.get("final")]
        if d.get("recovery") not in ("phase", "replay") or not finals or \
                not has_content(finals[-1].get("text", "")):
            bad.append(f"{where}: recovered done without a final delta with text: {frames}")
    else:
        if "recovered" in d or "recovery" in d:
            bad.append(f"{where}: unrecovered done carries recovery fields: {d}")
        if limit is not None and (secs is None or secs > limit):
            bad.append(f"{where}: plain done after {secs} s, over {limit:.3f} s")
    print(f"     {where}: EMPTY-prone turn -> {text_of(frames)!r}, recovered="
          f"{d.get('recovered', False)}/{d.get('recovery', '-')} in "
          f"{secs if secs is None else round(secs, 3)} s")


def cmd_recovery(a) -> int:
    bad = []
    a.log_dir = a.log_dir or os.environ.get("TMPDIR", "/tmp")
    full = load_pcm(a.clip)
    on = speech_onset(full)
    empty_prone = full[on:on + int(16000 * a.cut) * 2]
    healthy = load_pcm(a.healthy)
    noise = noise_pcm(1.0, 30.0, 7)          # about -60 dBFS
    silence = bytes(2 * int(16000 * 0.6))
    q = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
    t_out = a.timeout

    # ---- flag off: the server as it was
    with Server(a, a.port, []) as srv:
        st, _, body = http_request(a.host, srv.port, q + "&recover=1")
        if " 400 " not in st + " " or b"unknown_query_parameter" not in body:
            bad.append(f"off: recover=1 should be an unknown key, got {st!r}")
        run_turns(a.host, srv.port, q, [healthy], t_out)        # warm-up, not timed
        (off_empty, off_empty_s), = run_turns(a.host, srv.port, q, [empty_prone], t_out)
        (off_healthy, _), = run_turns(a.host, srv.port, q, [healthy], t_out)
        if "recovery" in health(a.host, srv.port):
            bad.append("off: /v1/health has a recovery block")
    done = off_empty[-1] if off_empty else {}
    if done.get("type") != "done" or "recovered" in done or "recovery" in done:
        bad.append(f"(a) off: expected a plain done, got {done}")
    empty_primary = not has_content(text_of(off_empty))
    if not empty_primary:
        print(f"     note: the primary decoded the {a.cut} s cut as {text_of(off_empty)!r};"
              " the EMPTY-turn cases then exercise nothing")
    print(f"     (a) off: EMPTY-prone turn {text_of(off_empty)!r}, plain done in "
          f"{off_empty_s if off_empty_s is None else round(off_empty_s, 3)} s")

    # ---- flag on, a deadline long enough for the lane to answer and the duty
    # admission out of the way (1000 x the chunk period), so the recovery path
    # itself runs on any host, a loaded or slow one included (a single stream
    # stepping slower than real time is a SATURATED worker, which the default
    # admission rightly skips): the accept / reject path, the ordering behind a
    # held turn, the opt-out
    extra = ["--empty-recovery", "--recovery-deadline-ms", str(a.long_deadline_ms),
             "--recovery-max-duty", "1000"]
    with Server(a, a.port + 1, extra) as srv:
        st, _, body = http_request(a.host, srv.port, q + "&recover=2")
        if " 400 " not in st + " " or b"invalid_recover" not in body:
            bad.append(f"on: recover=2 should be a 400 invalid_recover, got {st!r}")
        run_turns(a.host, srv.port, q, [healthy], t_out)        # warm-up
        # (c) a healthy turn: every frame identical but its timing fields
        (on_healthy, _), = run_turns(a.host, srv.port, q, [healthy], t_out)
        same = strip_timing(on_healthy) == strip_timing(off_healthy)
        if not same:
            bad.append(f"(c) healthy turn differs: {strip_timing(on_healthy)} != "
                       f"{strip_timing(off_healthy)}")
        print(f"     (c) on: healthy turn {text_of(on_healthy)!r}, {len(on_healthy)} frames, "
              f"identical to off but timing: {same}")
        # (b) noise and silence, each given a free lane, never yield text
        for name, pcm in (("noise", noise), ("silence", silence)):
            (fr, _), = run_turns(a.host, srv.port, q, [pcm], t_out)
            d = fr[-1] if fr else {}
            if d.get("type") != "done" or d.get("recovered") or has_content(text_of(fr)):
                bad.append(f"(b) {name}: {text_of(fr)!r} done={d}")
            print(f"     (b) on: {name} -> {text_of(fr)!r}, recovered={d.get('recovered', False)}")
        # (d) the EMPTY-prone turn, with a lane free
        before = health(a.host, srv.port)["recovery"]["attempted"]
        (t1, t1_s), = run_turns(a.host, srv.port, q, [empty_prone], t_out)
        check_empty_turn(t1, t1_s, None, bad, "(d) FORCED (admission off) on/long")
        if empty_primary and health(a.host, srv.port)["recovery"]["attempted"] != before + 1:
            bad.append("(d) the EMPTY turn did not reach a lane")
        # ordering: the next turn is sent WHILE the first one is held, and must
        # come out after its `done`, decoded on its own
        sess = Session(a.host, srv.port, q)
        try:
            send_clip(sess, empty_prone)
            sess.send_text({"type": "finalize"})
            held = wait_held(a, srv.port, 1, 15.0) if empty_primary else False
            send_clip(sess, healthy)
            sess.send_text({"type": "finalize"})
            f1, _ = turn_frames(sess, t_out)
            f2, _ = turn_frames(sess, t_out)
            sess.send_close()
        finally:
            sess.close()
        seqs = [f["seq"] for f in f1 + f2 if "seq" in f]
        if empty_primary and not held:
            bad.append("(e) never saw the first turn held")
        if not f1 or f1[-1].get("type") != "done" or not f2 or f2[-1].get("type") != "done":
            bad.append(f"(e) two dones expected: {f1} / {f2}")
        if text_of(f2) != text_of(off_healthy):
            bad.append(f"(e) the turn behind the held one: {text_of(f2)!r} != "
                       f"{text_of(off_healthy)!r}")
        if seqs != sorted(seqs):
            bad.append(f"(e) frames out of order across the two turns: {seqs}")
        print(f"     (e) on: held={held}, first {text_of(f1)!r}, then {text_of(f2)!r}")
        # recover=0: the session opted out, the off answer
        (o0, _), = run_turns(a.host, srv.port, q + "&recover=0", [empty_prone], t_out)
        if strip_timing(o0) != strip_timing(off_empty):
            bad.append(f"recover=0: expected the off frames, got {o0}")
        wait_held(a, srv.port, 0, 15.0)
        rv = health(a.host, srv.port).get("recovery")
        check_books(rv, bad, "FORCED on/long")
        # a run where every turn was skipped proves nothing about recovery
        if empty_primary and isinstance(rv, dict) and rv["attempted"] < 1:
            bad.append(f"FORCED on/long: no turn reached a lane (attempted 0): {rv}")

    # ---- flag on, the default deadline: recovered or a plain done in bound
    extra = ["--empty-recovery", "--recovery-deadline-ms", str(a.deadline_ms)]
    with Server(a, a.port + 2, extra) as srv:
        run_turns(a.host, srv.port, q, [healthy], t_out)        # warm-up
        (t1, t1_s), = run_turns(a.host, srv.port, q, [empty_prone], t_out)
        limit = (off_empty_s or 0.0) + a.deadline_ms / 1000.0 + a.slack
        check_empty_turn(t1, t1_s, limit, bad, f"(d) on/{a.deadline_ms} ms")
        time.sleep(0.2)
        check_books(health(a.host, srv.port).get("recovery"), bad, f"on/{a.deadline_ms} ms")

    for b in bad:
        print("     " + b)
    return verdict(not bad, "--empty-recovery: off is unchanged, noise never recovers, "
                            "healthy turns identical, EMPTY turn recovered or bounded, "
                            "the next turn waits for the held done")


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--lookahead", default="3")
    ap.add_argument("--timeout", type=float, default=60.0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("utterances")
    p.add_argument("--clips", nargs="+", required=True)
    p.add_argument("--reference", required=True)
    p.set_defaults(fn=cmd_utterances)

    p = sub.add_parser("control")
    p.add_argument("--clip", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--message", default="teleport")
    p.add_argument("--expect-code", default="unknown_control")
    p.set_defaults(fn=cmd_control)

    p = sub.add_parser("bad-lang")
    p.add_argument("--clip", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--bad", default="zz-ZZ")
    p.set_defaults(fn=cmd_bad_lang)

    p = sub.add_parser("http")
    p.add_argument("--path", required=True)
    p.add_argument("--expect-status", type=int, required=True)
    p.add_argument("--expect-body", nargs="*")
    p.add_argument("--expect-header", nargs="*")
    p.set_defaults(fn=cmd_http)

    p = sub.add_parser("hold")
    p.add_argument("--clip")
    p.add_argument("--hold", type=float, default=5.0)
    p.add_argument("--pace", type=float, default=1.0)
    p.add_argument("--ready-file")
    p.add_argument("--expect-code")
    p.set_defaults(fn=cmd_hold)

    p = sub.add_parser("idle")
    p.add_argument("--wait", type=float, default=8.0)
    p.set_defaults(fn=cmd_idle)

    p = sub.add_parser("ping")
    p.add_argument("--wait", type=float, default=3.0)
    p.add_argument("--min-pings", type=int, default=2)
    p.set_defaults(fn=cmd_ping)

    p = sub.add_parser("recovery")
    p.add_argument("--server-bin", default="./mynah-asr-server")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--quant", default="int8")
    p.add_argument("--clip", default="tests/audio/test_en.wav",
                   help="its first --cut seconds of speech are the EMPTY-prone turn")
    p.add_argument("--cut", type=float, default=0.4)
    p.add_argument("--healthy", default="tests/audio/test_en.wav")
    p.add_argument("--deadline-ms", type=int, default=400)
    p.add_argument("--long-deadline-ms", type=int, default=10000,
                   help="the second server's deadline: long enough for a lane to answer")
    p.add_argument("--slack", type=float, default=1.0,
                   help="seconds of scheduling slack on top of off-time + deadline")
    p.add_argument("--log-dir")
    p.set_defaults(fn=cmd_recovery)

    a = ap.parse_args()
    try:
        return a.fn(a)
    except RuntimeError as e:          # a refused handshake where one was expected to work
        print(f"FAIL {a.cmd}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
