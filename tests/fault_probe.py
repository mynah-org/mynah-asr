#!/usr/bin/env python3
"""Provoked failures against a live mynah-asr-server, each with its invariants (S12-20).

Python stdlib only, like ws_probe.py, whose wire helpers it reuses. Every case
breaks ONE thing on purpose and then checks what the server did about it, from
the server's own /v1/health and never from the client's opinion:

  - the slot comes back: `slots.active` returns to its baseline within a bound;
  - the right outcome counter moved, by exactly one, and no other did;
  - no zombie inference: the audio the model consumed AFTER the client went
    away (`audio_seconds` read just before the disconnect, and again once the
    server is quiet) is at most a step or two, never the rest of the ring;
  - the books balance: every session the server claimed is completed,
    cancelled, aborted before it started, or still active -- nothing else.

Cases (run by `suite`, in this order; each can also be run alone):

  rst-mid          25 s of audio blasted into the ring, then a TCP RST
  fin-mid          the same, then a plain close(): FIN with no WebSocket close
  rst-mid-silent   rst-mid with 28 s of silence: no delta is ever written, so
  fin-mid-silent   no write can fail -- only the input side can see the hangup
  close-then-rst   close frame (finalize) then an immediate RST
  close-then-rst-silent  the same with silence: only a hard-hangup check made
                   WHILE the tail is flushed can stop it
  half-close-ok    finalize, then shutdown(SHUT_WR): LEGAL, must still get `done`
  idle-open        a few frames, then silence with the socket open
  stall-mid-frame  a frame header and part of its payload, then silence
  oversize         a frame header over --max-frame-bytes
  rsv-bits         a frame with reserved bits set (RFC 6455 5.2: fail the connection)
  bad-control      a ping whose payload exceeds 125 bytes (RFC 6455 5.5)
  garbage          random bytes after the upgrade
  capacity         --cap streams held, the next one must be an HTTP 503
  neighbours       a healthy stream running while three others are killed
                   around it: its transcript must equal the unloaded reference
  abort-loop       N mixed aborts, then RSS growth and the balance
  worker-kill      (prefork only, not in `suite`) SIGKILL one worker under a live
                   stream: the router charges it to `lost`, its books balance,
                   the other worker's stream is untouched and new streams are served

Usage:
  python3 tests/fault_probe.py --port P --clip tests/audio/test_en.wav \
      --reference ref.json --cap 4 --server-pid PID suite
Exit 0 when every case holds, 1 otherwise; one OK/FAIL line per invariant.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ws_probe import (Session, collect_utterance, http_request,  # noqa: E402
                      load_pcm, send_clip, ws_frame)

# The model's default preset, so any streaming model will do. No `lang` at all
# for a model with no language prompt (the English-only EOU model refuses one).
STREAM_PATH = "/v1/audio/stream"
RATE = 16000


# ------------------------------------------------------------ the server's books

def health(port: int) -> dict:
    with urllib.request.urlopen(f"http://localhost:{port}/v1/health", timeout=5) as r:
        return json.loads(r.read())


def outcomes(h: dict) -> dict:
    """The terminal-outcome counters, flat: completed, aborted and one key per
    cancel bucket. Missing keys read as None so an old binary FAILS the balance
    instead of silently passing it."""
    out = {"sessions": h.get("sessions"), "completed": h.get("completed"),
           "aborted": h.get("aborted"), "active": (h.get("slots") or {}).get("active"),
           "cancelled": h.get("cancelled")}
    for k, v in (h.get("cancelled_by") or {}).items():
        out["cancel:" + k] = v
    return out


def wait_quiet(port: int, baseline_active: int = 0, timeout: float = 10.0):
    """Waits until the slot count is back at `baseline_active` and the fed-audio
    counter has stopped moving (two equal reads 300 ms apart). Returns
    (health, seconds waited) or (last health, None) on timeout."""
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout:
        h = health(port)
        act = (h.get("slots") or {}).get("active", -1)
        if act == baseline_active and last is not None and \
                h.get("audio_seconds") == last.get("audio_seconds"):
            return h, time.monotonic() - t0
        last = h
        time.sleep(0.3)
    return last, None


def balance(h: dict):
    """sessions == completed + cancelled + aborted + active. Returns (ok, text)."""
    o = outcomes(h)
    need = ("sessions", "completed", "aborted", "active", "cancelled")
    if any(o[k] is None for k in need):
        return False, "outcome counters missing: " + ", ".join(k for k in need if o[k] is None)
    rhs = o["completed"] + o["cancelled"] + o["aborted"] + o["active"]
    txt = (f"sessions={o['sessions']} completed={o['completed']} cancelled={o['cancelled']} "
           f"aborted={o['aborted']} active={o['active']}")
    return o["sessions"] == rhs, txt


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, ok: bool, what: str) -> bool:
        print(("OK   " if ok else "FAIL ") + what, flush=True)
        if not ok:
            self.failed += 1
        return ok


# ------------------------------------------------------------ raw clients

def raw_upgrade(port: int) -> socket.socket:
    """A socket past the 101 with nothing else attached: the cases below need to
    send bytes no well-behaved client would, so they cannot use Session."""
    sock = socket.create_connection(("localhost", port), timeout=10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    key = "dGhlIHNhbXBsZSBub25jZQ=="
    sock.sendall((f"GET {STREAM_PATH} HTTP/1.1\r\nHost: localhost:{port}\r\n"
                  f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        part = sock.recv(4096)
        if not part:
            break
        resp += part
    if b" 101 " not in resp.split(b"\r\n", 1)[0]:
        sock.close()
        raise RuntimeError("handshake refused: " + resp.split(b"\r\n", 1)[0].decode())
    return sock


def rst_close(sock: socket.socket) -> None:
    """SO_LINGER {on, 0}: close() sends a RST instead of a FIN, which is what a
    crashed client or a middlebox dropping the connection looks like."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    except OSError:
        pass        # the server already closed it (macOS: EINVAL); a close is all that is left
    sock.close()


def drain(sock: socket.socket, seconds: float) -> bytes:
    """Reads whatever the server sends for up to `seconds`, stopping at EOF."""
    sock.settimeout(0.2)
    got = b""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            part = sock.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            break
        if not part:
            break
        got += part
    return got


def error_codes(buf: bytes) -> list:
    """Error codes carried by the text frames in a raw server byte stream."""
    codes, i = [], 0
    while i + 2 <= len(buf):
        op, n = buf[i] & 0x0F, buf[i + 1] & 0x7F
        i += 2
        if n == 126:
            n = struct.unpack(">H", buf[i:i + 2])[0]; i += 2
        elif n == 127:
            n = struct.unpack(">Q", buf[i:i + 8])[0]; i += 8
        payload = buf[i:i + n]
        i += n
        if op == 0x1:
            try:
                m = json.loads(payload)
            except ValueError:
                continue
            if m.get("type") == "error":
                codes.append(m.get("code"))
    return codes


def long_pcm(clip: str, seconds: float) -> bytes:
    pcm = load_pcm(clip)
    want = int(seconds * RATE) * 2
    return (pcm * (want // len(pcm) + 1))[:want]


def blast(sock: socket.socket, pcm: bytes) -> None:
    step = RATE // 10 * 2
    for off in range(0, len(pcm), step):
        sock.sendall(ws_frame(0x2, pcm[off:off + step]))


def rss_kb(pid: int) -> int:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) if out else -1
    except (OSError, ValueError, subprocess.SubprocessError):
        return -1


# ------------------------------------------------------------ the cases

class Suite:
    def __init__(self, a):
        self.a = a
        self.c = Checks()
        self.pcm = load_pcm(a.clip)
        self.ref = json.load(open(a.reference, encoding="utf-8")).get(a.clip) if a.reference else None

    # -- shared shape: provoke, wait, check the books ----------------------
    def settle(self, name: str, before: dict, expect: dict, zombie_from=None,
               zombie_max: float = None):
        h, waited = wait_quiet(self.a.port, 0, self.a.settle)
        self.c.check(waited is not None,
                     f"{name}: slots back to 0 ({'%.1f s' % waited if waited is not None else 'TIMEOUT, active=%s' % (h.get('slots') or {}).get('active')})")
        ob, oa = outcomes(before), outcomes(h)
        moved = {k: (oa.get(k) or 0) - (ob.get(k) or 0) for k in set(oa) | set(ob)
                 if k not in ("sessions", "active", "cancelled")
                 and (oa.get(k) or 0) != (ob.get(k) or 0)}
        self.c.check(moved == expect, f"{name}: outcome counters moved {moved or '{}'} "
                                      f"(expected {expect})")
        if zombie_from is not None:
            z = h.get("audio_seconds", 0.0) - zombie_from
            self.c.check(z <= zombie_max,
                         f"{name}: audio fed after the client went away {z:.2f} s "
                         f"(bound {zombie_max:.2f} s)")
        ok, txt = balance(h)
        self.c.check(ok, f"{name}: books balance ({txt})")
        return h

    def abort_mid(self, name: str, how: str, silent: bool = False):
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        pushed = 28.0                        # under the 30 s ring: the push never blocks
        # Silence is the hard case: speech makes the server WRITE deltas, and a
        # write to a dead peer fails with EPIPE within a delta or two, so speech
        # gets caught by the output side whatever the input side does. Silence
        # writes nothing, and only the input side can notice the client left.
        pcm = bytes(int(pushed * RATE) * 2) if silent else long_pcm(self.a.clip, pushed)
        blast(sock, pcm)
        fed_at = health(self.a.port).get("audio_seconds", 0.0)
        if how == "rst":
            rst_close(sock)
        else:
            sock.close()
        # The case only means something if audio was still QUEUED when the
        # client left: on a fast machine the model may have eaten the whole
        # ring already, and then "nothing was fed afterwards" proves nothing.
        queued = pushed - (fed_at - before.get("audio_seconds", 0.0))
        self.c.check(queued >= 5.0, f"{name}: discriminating ({queued:.1f} s still queued "
                                    f"at the disconnect, need >= 5)")
        self.settle(name, before, {"cancel:peer_gone": 1}, fed_at, self.a.zombie_s)

    def case_rst_mid(self):
        self.abort_mid("rst-mid", "rst")

    def case_fin_mid(self):
        self.abort_mid("fin-mid", "fin")

    def case_rst_mid_silent(self):
        self.abort_mid("rst-mid-silent", "rst", silent=True)

    def case_fin_mid_silent(self):
        self.abort_mid("fin-mid-silent", "fin", silent=True)

    def case_close_then_rst_silent(self):
        # The same, with silence: the tail writes nothing, so no write can fail
        # and only a hard-hangup check DURING the finalize can see the RST. A
        # finalize is legal work, but 28 s of it for a reset peer is a zombie.
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        blast(sock, bytes(int(28.0 * RATE) * 2))
        sock.sendall(ws_frame(0x8, b""))
        fed_at = health(self.a.port).get("audio_seconds", 0.0)
        queued = 28.0 - (fed_at - before.get("audio_seconds", 0.0))
        rst_close(sock)
        self.c.check(queued >= 5.0, f"close-then-rst-silent: discriminating ({queued:.1f} s "
                                    f"still queued at the RST, need >= 5)")
        self.settle("close-then-rst-silent", before, {"cancel:peer_gone": 1}, fed_at,
                    self.a.zombie_s)

    def case_close_then_rst(self):
        # The client ASKED for the tail and then vanished. The work it asked for
        # may run, but only until the server learns the peer is gone -- the
        # bound is the ring (25 s) minus what was fed before, so this checks
        # detection happens at all, not that it beats the tail.
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        blast(sock, long_pcm(self.a.clip, 28.0))
        sock.sendall(ws_frame(0x8, b""))
        fed_at = health(self.a.port).get("audio_seconds", 0.0)
        rst_close(sock)
        h, waited = wait_quiet(self.a.port, 0, self.a.settle)
        z = h.get("audio_seconds", 0.0) - fed_at
        self.c.check(waited is not None, f"close-then-rst: slots back to 0")
        ob, oa = outcomes(before), outcomes(h)
        # Either outcome is correct here -- a tail that finished before the RST
        # was noticed is `completed` -- but it must be exactly ONE of them.
        d_done = (oa.get("completed") or 0) - (ob.get("completed") or 0)
        d_gone = (oa.get("cancel:peer_gone") or 0) - (ob.get("cancel:peer_gone") or 0)
        self.c.check(d_done + d_gone == 1,
                     f"close-then-rst: exactly one outcome (completed +{d_done}, "
                     f"peer_gone +{d_gone}); fed after the RST {z:.2f} s")
        ok, txt = balance(h)
        self.c.check(ok, f"close-then-rst: books balance ({txt})")

    def case_half_close_ok(self):
        # The legal shape the cancel-on-EOF rule must NOT break: the client asks
        # for the tail, then half-closes its sending side and waits for `done`.
        before = health(self.a.port)
        sess = Session("localhost", self.a.port, STREAM_PATH)
        send_clip(sess, self.pcm)
        sess.send_text({"type": "finalize"})
        sess.sock.shutdown(socket.SHUT_WR)
        text, _, errors = collect_utterance(sess, 60.0)
        sess.close()
        self.c.check(not errors, f"half-close-ok: no error frames ({errors})")
        if self.ref is not None:
            self.c.check(text == self.ref, "half-close-ok: transcript equals the reference")
        self.settle("half-close-ok", before, {"completed": 1})

    def case_idle_open(self):
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        blast(sock, self.pcm[: RATE * 2 // 2])
        got = drain(sock, self.a.idle_ms / 1000.0 + 3.0)
        sock.close()
        self.c.check("idle_timeout" in error_codes(got),
                     f"idle-open: client told idle_timeout ({error_codes(got)})")
        self.settle("idle-open", before, {"cancel:idle_timeout": 1})

    def case_stall_mid_frame(self):
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        blast(sock, self.pcm[: RATE])
        frame = ws_frame(0x2, b"\0" * 3200)
        sock.sendall(frame[:100])            # header + a sliver of the payload
        got = drain(sock, self.a.idle_ms / 1000.0 + 3.0)
        sock.close()
        self.c.check("idle_timeout" in error_codes(got),
                     f"stall-mid-frame: client told idle_timeout ({error_codes(got)})")
        self.settle("stall-mid-frame", before, {"cancel:idle_timeout": 1})

    def one_bad_frame(self, name: str, raw: bytes, code: str, bucket: str):
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        blast(sock, self.pcm[: RATE])
        sock.sendall(raw)
        got = drain(sock, 3.0)
        sock.close()
        self.c.check(code in error_codes(got), f"{name}: client told {code} ({error_codes(got)})")
        self.settle(name, before, {"cancel:" + bucket: 1})

    def case_oversize(self):
        n = self.a.max_frame_bytes + 1
        raw = bytes([0x82, 0x80 | 127]) + struct.pack(">Q", n) + b"\0\0\0\0"
        self.one_bad_frame("oversize", raw, "frame_too_large", "frame_too_large")

    def case_rsv_bits(self):
        raw = bytes([0x80 | 0x40 | 0x2, 0x80 | 4]) + b"\0\0\0\0" + b"\0\0\0\0"
        self.one_bad_frame("rsv-bits", raw, "protocol_error", "protocol_error")

    def case_bad_control(self):
        raw = bytes([0x89, 0x80 | 126]) + struct.pack(">H", 200) + b"\0\0\0\0" + b"x" * 200
        self.one_bad_frame("bad-control", raw, "protocol_error", "protocol_error")

    def case_garbage(self):
        before = health(self.a.port)
        sock = raw_upgrade(self.a.port)
        rnd = random.Random(20260925)
        # Unmasked, reserved bits, absurd lengths: whatever the parser makes of
        # it, the session must END, with one outcome, and the slot come back.
        sock.sendall(bytes(rnd.getrandbits(8) for _ in range(64)))
        drain(sock, 3.0)
        rst_close(sock)
        h, waited = wait_quiet(self.a.port, 0, self.a.settle)
        self.c.check(waited is not None, "garbage: slots back to 0")
        ok, txt = balance(h)
        self.c.check(ok, f"garbage: books balance ({txt})")
        self.c.check((outcomes(h).get("sessions") or 0) - (outcomes(before).get("sessions") or 0) == 1,
                     "garbage: counted as one session")

    def case_capacity(self):
        before = health(self.a.port)
        held = []
        try:
            for _ in range(self.a.cap):
                held.append(raw_upgrade(self.a.port))
            status, headers, body = http_request("localhost", self.a.port, STREAM_PATH)
            self.c.check(" 503 " in status and b"server_at_capacity" in body,
                         f"capacity: stream {self.a.cap + 1} refused with 503 ({status})")
            mid = health(self.a.port)
            self.c.check(outcomes(mid)["sessions"] - outcomes(before)["sessions"] == self.a.cap,
                         "capacity: the refusal is not a session")
        finally:
            for s in held:
                s.sendall(ws_frame(0x8, b""))
                drain(s, 0.5)
                s.close()
        self.settle("capacity", before, {"completed": self.a.cap})

    def case_neighbours(self):
        # A healthy paced stream while three others die around it. The point is
        # isolation: its words must not depend on who else crashed.
        before = health(self.a.port)
        result = {}

        def healthy():
            sess = Session("localhost", self.a.port, STREAM_PATH)
            send_clip(sess, self.pcm, pace=2.0)
            sess.send_close()
            result["text"], _, result["errors"] = collect_utterance(sess, 60.0)
            sess.close()

        t = threading.Thread(target=healthy)
        t.start()
        victims = []
        for how in ("rst", "fin", "garbage"):
            s = raw_upgrade(self.a.port)
            blast(s, long_pcm(self.a.clip, 10.0))
            victims.append((how, s))
        time.sleep(0.5)
        for how, s in victims:
            if how == "rst":
                rst_close(s)
            elif how == "fin":
                s.close()
            else:
                s.sendall(bytes([0xF2, 0x80 | 4]) + b"\0" * 8)
                rst_close(s)
        t.join(90)
        self.c.check(not result.get("errors"), f"neighbours: healthy stream saw no error "
                                               f"({result.get('errors')})")
        if self.ref is not None:
            self.c.check(result.get("text") == self.ref,
                         "neighbours: healthy transcript equals the unloaded reference")
        h, waited = wait_quiet(self.a.port, 0, self.a.settle)
        self.c.check(waited is not None, "neighbours: slots back to 0")
        ok, txt = balance(h)
        self.c.check(ok, f"neighbours: books balance ({txt})")
        d = {k: (outcomes(h).get(k) or 0) - (outcomes(before).get(k) or 0)
             for k in ("completed", "cancel:peer_gone", "cancel:protocol_error")}
        self.c.check(d == {"completed": 1, "cancel:peer_gone": 2, "cancel:protocol_error": 1},
                     f"neighbours: one completed, three cancelled by cause ({d})")

    def case_abort_loop(self):
        # RSS after a warm-up round, then N more: growth must be flat, not
        # proportional to the aborts.
        def one_round(k):
            how = ("rst", "fin", "idle-close", "garbage")[k % 4]
            s = raw_upgrade(self.a.port)
            blast(s, long_pcm(self.a.clip, 3.0 + (k % 5)))
            time.sleep(0.05 * (k % 7))
            if how == "rst":
                rst_close(s)
            elif how == "fin":
                s.close()
            elif how == "idle-close":
                s.sendall(ws_frame(0x8, b""))
                drain(s, 2.0)
                s.close()
            else:
                s.sendall(bytes([0xF2, 0x80 | 4]) + b"\0" * 8)
                rst_close(s)

        for k in range(8):
            one_round(k)
        wait_quiet(self.a.port, 0, self.a.settle)
        rss0 = rss_kb(self.a.server_pid) if self.a.server_pid else -1
        for k in range(self.a.loops):
            one_round(k)
        h, waited = wait_quiet(self.a.port, 0, self.a.settle)
        rss1 = rss_kb(self.a.server_pid) if self.a.server_pid else -1
        self.c.check(waited is not None, f"abort-loop: slots back to 0 after {self.a.loops} aborts")
        ok, txt = balance(h)
        self.c.check(ok, f"abort-loop: books balance ({txt})")
        if rss0 > 0 and rss1 > 0:
            grow = (rss1 - rss0) / 1024.0
            self.c.check(grow <= self.a.rss_mb,
                         f"abort-loop: RSS {rss0 / 1024:.0f} -> {rss1 / 1024:.0f} MB "
                         f"(+{grow:.1f}, bound {self.a.rss_mb:.0f})")

    # -- prefork: a worker process dies under a live stream -----------------
    def router_metrics(self) -> dict:
        """The router's per-worker series: {worker label: {metric: value}}."""
        with urllib.request.urlopen(f"http://localhost:{self.a.metrics_port}/metrics",
                                    timeout=5) as r:
            text = r.read().decode()
        out = {}
        for line in text.splitlines():
            if line.startswith("#") or "{" not in line:
                continue
            name, _, rest = line.partition("{")
            labels, _, val = rest.rpartition("} ")
            w = [kv.split("=", 1)[1].strip('"') for kv in labels.split(",")
                 if kv.startswith("worker=")]
            if w and name.startswith("mynah_asr_worker_"):
                out.setdefault(w[0], {})[name[len("mynah_asr_worker_"):]] = float(val)
        return out

    def case_worker_kill(self):
        kids = subprocess.run(["pgrep", "-P", str(self.a.server_pid)], capture_output=True,
                              text=True).stdout.split()
        self.c.check(len(kids) >= 2, f"worker-kill: {len(kids)} worker processes found")
        if len(kids) < 2:
            return
        # Two paced streams: least-loaded routing puts one on each worker.
        results = [{}, {}]

        def run(slot):
            try:
                sess = Session("localhost", self.a.port, STREAM_PATH)
            except Exception as e:  # noqa: BLE001
                results[slot]["errors"] = [f"connect {e}"]
                return
            try:
                send_clip(sess, long_pcm(self.a.clip, 8.0), pace=1.0)
                sess.send_close()
            except OSError:
                pass
            results[slot]["text"], _, results[slot]["errors"] = collect_utterance(sess, 30.0)
            results[slot]["closed"] = sess.peer_closed.is_set()
            sess.close()

        ts = [threading.Thread(target=run, args=(k,)) for k in (0, 1)]
        for t in ts:
            t.start()
        time.sleep(2.0)
        m0 = self.router_metrics()
        busy = [w for w, v in m0.items() if v.get("inflight", 0) > 0]
        self.c.check(len(busy) == 2, f"worker-kill: both workers hold a stream ({m0})")
        victim = kids[0]
        os.kill(int(victim), 9)
        for t in ts:
            t.join(60)
        m1 = self.router_metrics()
        dead = [w for w, v in m1.items() if v.get("up", 1) == 0]
        self.c.check(len(dead) == 1, f"worker-kill: the router sees exactly one worker down ({dead})")
        lost = sum(v.get("lost_total", 0) for v in m1.values())
        self.c.check(lost == 1, f"worker-kill: one connection charged to `lost` ({lost})")
        for w, v in m1.items():
            lhs = v.get("assigned_total", 0)
            rhs = v.get("completed_total", 0) + v.get("lost_total", 0) + v.get("inflight", 0)
            self.c.check(lhs == rhs, f"worker-kill: worker {w} books balance "
                                     f"(assigned {lhs:.0f} = completed + lost + inflight {rhs:.0f})")
        good = [r for r in results if not r.get("errors")]
        bad = [r for r in results if r.get("errors")]
        self.c.check(len(good) == 1 and len(bad) == 1,
                     f"worker-kill: one stream finished, one saw the loss "
                     f"({[r.get('errors') for r in results]})")
        # capacity after the death: a new stream is served by the survivor
        sess = Session("localhost", self.a.port, STREAM_PATH)
        send_clip(sess, self.pcm)
        sess.send_close()
        text, _, errors = collect_utterance(sess, 30.0)
        sess.close()
        self.c.check(not errors and (self.ref is None or text == self.ref),
                     f"worker-kill: a new stream after the death is served correctly ({errors})")

    CASES = ["rst-mid", "fin-mid", "rst-mid-silent", "fin-mid-silent", "close-then-rst", "close-then-rst-silent", "half-close-ok", "idle-open",
             "stall-mid-frame", "oversize", "rsv-bits", "bad-control", "garbage",
             "capacity", "neighbours", "abort-loop"]

    def run(self, name: str) -> None:
        fn = getattr(self, "case_" + name.replace("-", "_"))
        print(f"---- {name}", flush=True)
        try:
            fn()
        except Exception as e:  # a crash in a case is a FAIL of that case, not of the suite
            self.c.check(False, f"{name}: raised {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--clip", default="tests/audio/test_en.wav")
    ap.add_argument("--reference", default=None, help="JSON {clip: transcript}")
    ap.add_argument("--lang", default="auto", help="the `lang` query value; empty = none")
    ap.add_argument("--cap", type=int, default=4, help="the server's --cap")
    ap.add_argument("--idle-ms", type=int, default=2000, help="the server's --idle-ms")
    ap.add_argument("--max-frame-bytes", type=int, default=65536,
                    help="the server's --max-frame-bytes")
    ap.add_argument("--server-pid", type=int, default=0, help="for the RSS check")
    ap.add_argument("--zombie-s", type=float, default=2.0,
                    help="audio the model may consume after a hangup (s)")
    ap.add_argument("--settle", type=float, default=15.0, help="bound on slot recovery (s)")
    ap.add_argument("--loops", type=int, default=40)
    ap.add_argument("--metrics-port", type=int, default=0, help="the prefork router's /metrics")
    ap.add_argument("--rss-mb", type=float, default=24.0)
    ap.add_argument("cases", nargs="+", help="'suite' or case names: " + " ".join(Suite.CASES))
    a = ap.parse_args()
    global STREAM_PATH
    if a.lang:
        STREAM_PATH += "?lang=" + a.lang
    s = Suite(a)
    names = Suite.CASES if a.cases == ["suite"] else a.cases
    for n in names:
        s.run(n)
    if names != ["worker-kill"]:
        h = health(a.port)
        ok, txt = balance(h)
        s.c.check(ok, f"final: books balance ({txt})")
    print(f"fault-probe: {s.c.failed} failed invariant(s)")
    return 1 if s.c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
