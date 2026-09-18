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

Usage: python3 tests/ws_probe.py <command> [options]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import socket
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


def verdict(ok: bool, what: str) -> int:
    print(("OK   " if ok else "FAIL ") + what)
    return 0 if ok else 1


# ------------------------------------------------------------- commands

def cmd_utterances(a) -> int:
    ref = load_reference(a.reference)
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
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
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
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
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
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
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
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
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
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
    path = f"/v1/audio/stream?lang={a.lang}&lookahead={a.lookahead}"
    sess = Session(a.host, a.port, path)
    try:
        time.sleep(a.wait)
        seen = sess.pings
    finally:
        sess.close()
    print(f"     {seen} ping(s) in {a.wait} s")
    return verdict(seen >= a.min_pings,
                   f"the server pings (>= {a.min_pings} in {a.wait} s)")


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

    a = ap.parse_args()
    try:
        return a.fn(a)
    except RuntimeError as e:          # a refused handshake where one was expected to work
        print(f"FAIL {a.cmd}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
