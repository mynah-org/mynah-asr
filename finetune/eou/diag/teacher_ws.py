#!/usr/bin/env python3
"""Teacher transcripts for a whole manifest through mynah-asr-server-cuda (WebSocket, GPU),
annotated per utterance: the label-quality score used to filter a training source.

    ./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 &
    python3 teacher_ws.py --manifest train_vp.json --out train_vp.teacher.json --port 8295 --lang it --conc 128
        [--check teacher_train_vp.json]      # agreement with CLI teacher transcripts of the same clips

Each clip is streamed unpaced (1 s frames), then closed; the transcript is the
concatenation of the server's `delta` texts (tools/bench/streaming_metrics.py). The
output manifest keeps every input field and adds `teacher` (normalised text) and
`teacher_wer` (accent-insensitive WER of the teacher against the manifest text, %).
Filtering is a separate, cheap step on that file (any threshold, no re-run).
Standard library only; the WebSocket framing is tools/eval/ws_client.py's.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import socket
import sys
import time
import unicodedata
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from eval.ws_client import ws_recv, ws_send  # noqa: E402


def norm(t):
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())


def lev(x, y):
    p = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        c = [i]
        for j, v in enumerate(y, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]


def transcribe(path, host, port, lang, tries=20):
    w = wave.open(path)
    pcm = w.readframes(w.getnframes())
    for k in range(tries):
        try:
            s = socket.create_connection((host, port), timeout=120)
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall(f"GET /v1/audio/stream?lang={lang} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
            resp = b""
            while b"\r\n\r\n" not in resp:
                resp += s.recv(4096)
            if b" 101" not in resp.split(b"\r\n")[0]:
                s.close(); time.sleep(0.5 + 0.2 * k); continue      # 503 at capacity: retry
            for off in range(0, len(pcm), 32000):
                ws_send(s, 0x2, pcm[off:off + 32000])
            ws_send(s, 0x8, b"")
            texts = []
            while True:
                op, payload = ws_recv(s)
                if op == 0x8:
                    break
                if op != 0x1:
                    continue
                msg = json.loads(payload)
                kind = msg.get("type") or ("done" if msg.get("done") else "delta")
                if kind == "delta" and msg.get("text"):
                    texts.append(msg["text"])
                if kind in ("done", "error"):
                    break
            s.close()
            return "".join(texts)
        except (OSError, ConnectionError, ValueError):
            time.sleep(0.5 + 0.2 * k)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8295)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", default="", help="teacher_check.py JSON of the same clips: agreement report")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest)]
    if a.check:
        want = {x["audio_filepath"] for x in json.load(open(a.check))}
        rows = [r for r in rows if r["audio_filepath"] in want]
    if a.limit:
        rows = rows[: a.limit]
    t0 = time.time(); done = 0; fails = 0

    def one(r):
        return r, transcribe(r["audio_filepath"], a.host, a.port, a.lang)

    with cf.ThreadPoolExecutor(a.conc) as ex, open(a.out + ".tmp", "w") as f:
        for r, t in ex.map(one, rows):
            done += 1
            if t is None:
                fails += 1
                continue
            ref = norm(r["text"])
            r["teacher"] = norm(t)
            r["teacher_wer"] = round(100 * lev(ref.split(), r["teacher"].split()) / max(1, len(ref.split())), 1)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if done % 1000 == 0:
                aud = sum(x["duration"] for x in rows[:done])
                print(f"  {done}/{len(rows)} clips, {aud / (time.time() - t0):.0f}x RT, fails {fails}", flush=True)
    os.replace(a.out + ".tmp", a.out)
    out = [json.loads(l) for l in open(a.out)]
    w = sorted(x["teacher_wer"] for x in out)
    pct = lambda q: w[min(len(w) - 1, int(q * len(w)))] if w else None  # noqa: E731
    print(f"== {a.manifest}: {len(out)} annotated, {fails} failed, {time.time() - t0:.0f}s; teacher WER p50 {pct(.5)} "
          f"p75 {pct(.75)} p90 {pct(.9)}; <=30 %: {sum(v <= 30 for v in w)}  <=50 %: {sum(v <= 50 for v in w)}  "
          f"<=80 %: {sum(v <= 80 for v in w)}", flush=True)
    if a.check:
        cli = {x["audio_filepath"]: x["teacher"] for x in json.load(open(a.check))}
        same = sum(cli.get(x["audio_filepath"]) == x["teacher"] for x in out)
        d = [lev(cli[x["audio_filepath"]].split(), x["teacher"].split()) / max(1, len(cli[x["audio_filepath"]].split()))
             for x in out if x["audio_filepath"] in cli]
        print(f"== check vs CLI teacher: identical {same}/{len(out)}, mean word distance {100 * sum(d) / max(1, len(d)):.2f} %")


if __name__ == "__main__":
    main()
