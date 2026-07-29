#!/usr/bin/env python3
"""Client WebSocket minimale (stdlib) per testare /v1/audio/stream di mynah-asr-server.

Uso: uv run python -m eval.ws_client <file.wav> [host=localhost] [port=8090]
                                     [lang=auto] [lookahead=3]
Invia il PCM in chunk da 100 ms come frame binari e stampa i delta JSON ricevuti.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import sys
import wave


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
                raise ConnectionError("connessione chiusa")
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


def main() -> None:
    wav_path = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8090
    lang = sys.argv[4] if len(sys.argv) > 4 else "auto"
    lookahead = sys.argv[5] if len(sys.argv) > 5 else "3"

    w = wave.open(wav_path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, "serve WAV 16 kHz mono"
    pcm = w.readframes(w.getnframes())

    sock = socket.create_connection((host, port), timeout=60)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall(
        f"GET /v1/audio/stream?lang={lang}&lookahead={lookahead} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    assert b"101" in resp.split(b"\r\n")[0], f"handshake fallito: {resp[:100]!r}"

    sock.settimeout(0.05)
    chunk_bytes = 3200  # 100 ms di s16le a 16 kHz

    def drain() -> bool:
        try:
            opcode, payload = ws_recv(sock)
        except (TimeoutError, socket.timeout):
            return True
        if opcode == 0x8:
            return False
        if opcode == 0x1:
            msg = json.loads(payload)
            if msg.get("done"):
                print(f"\n[done, language={msg.get('language')}]")
                return False
            print(msg.get("text", ""), end="", flush=True)
        return True

    for off in range(0, len(pcm), chunk_bytes):
        ws_send(sock, 0x2, pcm[off:off + chunk_bytes])
        drain()

    ws_send(sock, 0x8, b"")  # close: il server processa la coda e risponde
    sock.settimeout(30)
    while drain():
        pass
    sock.close()


if __name__ == "__main__":
    main()
