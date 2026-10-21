#!/usr/bin/env python3
"""NeMo (s2_reset.py) vs Mynah, clip by clip, on the exact bytes Mynah is fed (no training, no selection).

    prep:    mynah_parity.py prep --vals "mls=a.json,..." --per-set 200 --post-pad-s 2.0 --work W
             -> W/clips.jsonl + W/pad_<set>.json: the s2_reset.py clip selection (same stride), each clip with the
                same 2 s of -75 dB quiet (RandomState(len(x))) appended and written as 16-bit PCM WAV (Mynah reads
                PCM16 only; the original samples are bit-identical, only the quiet tail is rounded). The NeMo side
                is s2_reset.py --post-pad-s 0 --dump on W/pad_<set>.json, i.e. the same bytes.
    cli:     mynah_parity.py cli --work W --pack P --bin ./mynah-asr [--jobs 48] --out W/cli.jsonl
             (CPU engine, `mynah-asr stream --deltas`, f32)
    ws:      mynah_parity.py ws --work W --port 8291 [--conc 16] [--realtime 0] --out W/ws.jsonl
             (the CUDA server's /v1/audio/stream, 100 ms s16le frames; --realtime 1 paces them at 1x)
    compare: mynah_parity.py compare --work W --nemo-dump D.jsonl --runs cli=W/cli.jsonl,ws=W/ws.jsonl --lang fr
             --out R.json

Every protocol is scored by ONE function with the s2_reset.py definitions on the ORIGINAL clip (energy speech end,
80 ms frames): reset-WER / CER / empty, FE = an <EOU> with >= 0.5 s of speech left, FE_any = an <EOU> followed by
>= 2 speech frames, recall = a first <EOU> at / after speech end - 0.5 s, latency = that <EOU> - speech end.
Time convention: NeMo <EOU> frame f -> f * 0.08 s (s2_reset.py); Mynah reports the END of the emitting encoder frame,
(f + 1) * 0.08 s, so f = round(t / 0.08) - 1; the observed offset is printed, not assumed.
Parity is clip level: normalised transcript equal, <EOU> count equal, <EOU> frames equal; divergent clips are listed.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))
import textnorm  # noqa: E402

FR = 0.08


def speech_mask(x):
    n = int(FR * 16000)
    f = x[: len(x) // n * n].reshape(-1, n)
    db = 10 * np.log10((f ** 2).mean(1) + 1e-10)
    return db > db.max() - 35


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def prep(a):
    import soundfile as sf
    W = Path(a.work); (W / "wav").mkdir(parents=True, exist_ok=True)
    out = open(W / "clips.jsonl", "w")
    for spec in a.vals.split(","):
        sname, p = spec.split("=", 1)
        rows = [json.loads(l) for l in open(p)]
        rows = rows[:: max(1, len(rows) // a.per_set)][: a.per_set]
        man = open(W / f"pad_{sname}.json", "w")
        for i, r in enumerate(rows):
            x, sr = sf.read(r["audio_filepath"], dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            assert sr == 16000, r["audio_filepath"]
            sm = speech_mask(x)
            on = np.nonzero(sm)[0]
            end = (on[-1] + 1) * FR if len(on) else len(x) / 16000
            if a.post_pad_s:
                x = np.concatenate([x, (np.random.RandomState(len(x)).randn(int(a.post_pad_s * 16000))
                                        * 10 ** (-75 / 20)).astype(np.float32)])
            w = W / "wav" / f"{sname}_{i:04d}.wav"
            sf.write(w, np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16), 16000, subtype="PCM_16")
            man.write(json.dumps({"audio_filepath": str(w), "duration": len(x) / 16000, "text": r["text"]},
                                 ensure_ascii=False) + "\n")
            out.write(json.dumps({"set": sname, "i": i, "orig": r["audio_filepath"], "wav": str(w), "text": r["text"],
                                  "speech_end_s": round(float(end), 3), "sm": "".join("1" if v else "0" for v in sm)},
                                 ensure_ascii=False) + "\n")
        man.close()
        print(f"== prep {sname}: {len(rows)} clips", flush=True)


def clips(work):
    return [json.loads(l) for l in open(Path(work) / "clips.jsonl")]


def cli(a):
    def one(c):
        p = subprocess.run([a.bin, "stream", "-m", a.pack, "-i", c["wav"], "--quant", "f32", "--deltas"],
                           capture_output=True, text=True, timeout=600, env={**os.environ, "MYNAH_ASR_THREADS": "1"})
        text, eous, final, segs = "", [], None, [""]
        for l in p.stdout.splitlines():
            if not l.startswith("{"):
                continue
            o = json.loads(l)
            if o.get("type") == "delta":
                text += o.get("text", ""); segs[-1] += o.get("text", "")
            elif o.get("type") == "eou" and o.get("source") == "model":
                eous.append(o["t"]); segs.append("")
            elif o.get("type") == "final":
                final = o
        return {"wav": c["wav"], "text": text, "eou_t": eous, "segs": segs, "rc": p.returncode,
                "final_eq": final is not None and final.get("text") == text}
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, clips(a.work)))
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"== cli {len(res)} clips in {time.time() - t0:.0f} s, rc!=0: {sum(r['rc'] != 0 for r in res)}, "
          f"final==deltas: {sum(r['final_eq'] for r in res)}", flush=True)


def ws_send(sock, opcode, payload):
    mask = os.urandom(4)
    n = len(payload)
    h = bytes([0x80 | opcode]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n)
                                  if n < 65536 else bytes([0x80 | 127]) + struct.pack(">Q", n))
    sock.sendall(h + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


def ws_recv(sock):
    def rd(n):
        b = b""
        while len(b) < n:
            p = sock.recv(n - len(b))
            if not p:
                raise ConnectionError("closed")
            b += p
        return b
    h = rd(2)
    n = h[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", rd(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", rd(8))[0]
    return h[0] & 0x0F, rd(n) if n else b""


def ws_one(c, port, realtime):
    import wave
    w = wave.open(c["wav"]); pcm = w.readframes(w.getnframes()); w.close()
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    s.sendall((f"GET /v1/audio/stream HTTP/1.1\r\nHost: localhost:{port}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {base64.b64encode(os.urandom(16)).decode()}\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(4096)
    assert b"101" in resp.split(b"\r\n")[0], resp[:120]
    text, eous, lags, err, segs = "", [], [], None, [""]

    def handle(op, pl):
        nonlocal text, err, segs
        if op == 0x8:
            return False
        if op != 0x1:
            return True
        m = json.loads(pl); k = m.get("type")
        if k == "delta":
            text += m.get("text", ""); segs[-1] += m.get("text", ""); lags.append(m.get("lag_ms", 0.0))
        elif k == "eou" and m.get("source") == "model":
            eous.append(m["t"]); segs.append("")
        elif k == "error":
            err = m.get("code"); return False
        elif k == "done" or m.get("done"):
            return False
        return True
    t0 = time.time(); s.settimeout(0.0005 if not realtime else 0.02)
    for j, off in enumerate(range(0, len(pcm), 3200)):
        if realtime:
            d = t0 + j * 0.1 - time.time()
            if d > 0:
                time.sleep(d)
        ws_send(s, 0x2, pcm[off:off + 3200])
        try:
            while True:
                op, pl = ws_recv(s)
                if not handle(op, pl):
                    break
        except (TimeoutError, socket.timeout, BlockingIOError):
            pass
    ws_send(s, 0x8, b""); s.settimeout(60)
    try:
        while handle(*ws_recv(s)):
            pass
    except ConnectionError:
        pass
    s.close()
    return {"wav": c["wav"], "text": text, "eou_t": eous, "segs": segs, "err": err, "lag_ms_p50": float(np.median(lags)) if lags else None,
            "lag_ms_max": float(max(lags)) if lags else None, "wall_s": round(time.time() - t0, 3)}


def ws(a):
    cs = clips(a.work)
    if a.limit:
        cs = cs[: a.limit]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(lambda c: ws_one(c, a.port, a.realtime), cs))
    with open(a.out, "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lag = [r["lag_ms_max"] for r in res if r["lag_ms_max"] is not None]
    print(f"== ws conc={a.conc} realtime={a.realtime}: {len(res)} clips in {time.time() - t0:.0f} s, errors "
          f"{sum(r['err'] is not None for r in res)}, delta lag_ms max p50/p99 "
          f"{np.percentile(lag, 50):.1f}/{np.percentile(lag, 99):.1f}" if lag else "", flush=True)


def score(c, hyp, eou_f, lang):
    ref = textnorm.norm(c["text"], lang).split()
    h = hyp.split()
    end = c["speech_end_s"]; sm = c["sm"]
    prem = [f for f in eou_f if f * FR < end - 0.5]
    true = [f * FR for f in eou_f if f * FR >= end - 0.5]
    return {"err": lev(ref, h), "N": len(ref), "cerr": lev(" ".join(ref), " ".join(h)), "cN": len(" ".join(ref)),
            "empty": not h, "fe": bool(prem), "fe_any": any(sm[f + 1:].count("1") >= 2 for f in eou_f),
            "true": bool(true), "lat": (true[0] - end) if true else None, "n_eou": len(eou_f)}


def agg(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}) + ["MACRO"]:
        rs = rows if s == "MACRO" else [r for r in rows if r["set"] == s]
        if s == "MACRO":
            per = [out[k]["wer"] for k in out]
            pc = [out[k]["cer"] for k in out]
        lat = [r["lat"] for r in rs if r["lat"] is not None]
        n = len(rs)
        o = {"clips": n, "wer": round(100 * sum(r["err"] for r in rs) / sum(r["N"] for r in rs), 2),
             "cer": round(100 * sum(r["cerr"] for r in rs) / sum(r["cN"] for r in rs), 2),
             "empty": sum(r["empty"] for r in rs), "FE%": round(100 * sum(r["fe"] for r in rs) / n, 2),
             "FE_any%": round(100 * sum(r["fe_any"] for r in rs) / n, 2),
             "recall%": round(100 * sum(r["true"] for r in rs) / n, 2),
             "lat_p50_p90": [round(float(np.percentile(lat, q)), 2) for q in (50, 90)] if lat else None,
             "eou_per_clip": round(sum(r["n_eou"] for r in rs) / n, 3)}
        if s == "MACRO":
            o["wer_macro"] = round(float(np.mean(per)), 2); o["cer_macro"] = round(float(np.mean(pc)), 2)
        out[s] = o
    return out


def nemo_segs(ids, pieces):
    """s2_reset ids -> segment texts split at <EOU> (1024): SentencePiece pieces joined, U+2581 -> space."""
    segs, cur = [], []
    for t in ids:
        if t == 1024:
            segs.append(cur); cur = []
        elif t < 1024:
            cur.append(pieces[t])
    segs.append(cur)
    return ["".join(p).replace("\u2581", " ") for p in segs]


def divergence_class(d, m, mf, lang, pieces):
    """PRE_EOU: the clips differ before / at the first <EOU> (first-segment text, or first <EOU> frame, or one side has
    no <EOU> where the other has one): encoder / decoder numerics, independent of the reset semantics.
    POST_RESET: first segment and first <EOU> frame identical, the difference is after the first reset (where Mynah
    also empties the encoder cache and s2_reset.py does not). NO_EOU: neither side emitted an <EOU>."""
    ns = [textnorm.norm(x, lang) for x in nemo_segs(d["ids"], pieces)]
    ms = [textnorm.norm(x, lang) for x in m.get("segs", [m["text"]])]
    nf = d["eou_frames"]
    if not nf and not mf:
        return "NO_EOU", ns, ms
    if not nf or not mf or nf[0] != mf[0] or ns[0] != ms[0]:
        return "PRE_EOU", ns, ms
    return "POST_RESET", ns, ms


def compare(a):
    cs = {c["wav"]: c for c in clips(a.work)}
    pieces = json.load(open(Path(a.pack) / "tokens.json"))
    nemo = {}
    for l in open(a.nemo_dump):
        d = json.loads(l)
        if d["protocol"] == "s2_reset":
            nemo[d["audio"]] = d
    assert set(nemo) == set(cs), (len(nemo), len(cs))
    report = {"nemo": None, "runs": {}}
    nrows = []
    for w, c in cs.items():
        d = nemo[w]
        nrows.append({"set": c["set"], **score(c, d["hyp"], d["eou_frames"], a.lang)})
    report["nemo"] = agg(nrows)
    print("== NeMo s2_reset (same bytes)", json.dumps(report["nemo"]["MACRO"]), flush=True)
    for spec in a.runs.split(","):
        name, p = spec.split("=", 1)
        mr = {json.loads(l)["wav"]: json.loads(l) for l in open(p)}
        rows, div, offs = [], [], []
        eq_txt = eq_n = eq_f = 0
        hh_err = hh_N = 0
        for w, c in cs.items():
            m, d = mr[w], nemo[w]
            hyp = textnorm.norm(m["text"], a.lang)
            mf = [int(round(t / FR)) - 1 for t in m["eou_t"]]
            rows.append({"set": c["set"], **score(c, hyp, mf, a.lang)})
            if len(mf) == len(d["eou_frames"]):
                offs += [t / FR - f for t, f in zip(m["eou_t"], d["eou_frames"])]
            t_ok, n_ok, f_ok = hyp == d["hyp"], len(mf) == len(d["eou_frames"]), mf == d["eou_frames"]
            eq_txt += t_ok; eq_n += n_ok; eq_f += f_ok
            hh_err += lev(d["hyp"].split(), hyp.split()); hh_N += max(1, len(d["hyp"].split()))
            if not (t_ok and f_ok):
                cls, ns, ms = divergence_class(d, m, mf, a.lang, pieces)
                div.append({"class": cls, "nemo_segs": ns, "mynah_segs": ms,"set": c["set"], "wav": w, "nemo_hyp": d["hyp"], "mynah_hyp": hyp, "nemo_eou_f": d["eou_frames"],
                            "mynah_eou_f": mf, "mynah_eou_t": m["eou_t"], "speech_end_s": c["speech_end_s"],
                            "nemo_premature": d["premature"], "text_equal": t_ok, "eou_frames_equal": f_ok})
        n = len(cs)
        r = {"metrics": agg(rows), "clips": n, "text_equal": eq_txt, "eou_count_equal": eq_n, "eou_frames_equal": eq_f,
             "hyp_vs_hyp_wer%": round(100 * hh_err / hh_N, 3),
             "eou_t_over_frame_minus_nemo_frame": {str(k): int(v) for k, v in
                                                   zip(*np.unique(np.round(offs, 3), return_counts=True))} if offs else {},
             "divergent_by_class": {k: sum(x["class"] == k for x in div) for k in ("PRE_EOU", "POST_RESET", "NO_EOU")},
             "divergent_with_nemo_premature": sum(bool(x["nemo_premature"]) for x in div),
             "divergent": div}
        report["runs"][name] = r
        print(f"== {name}: text equal {eq_txt}/{n}, eou count equal {eq_n}/{n}, eou frames equal {eq_f}/{n}, "
              f"hyp-vs-hyp WER {r['hyp_vs_hyp_wer%']} %, t/0.08 - f {r['eou_t_over_frame_minus_nemo_frame']}, "
              f"divergent by class {r['divergent_by_class']} (with a NeMo premature <EOU>: {r['divergent_with_nemo_premature']})", flush=True)
        print(f"== {name} metrics MACRO", json.dumps(r["metrics"]["MACRO"]), flush=True)
        for s in sorted(r["metrics"]):
            if s != "MACRO":
                print(f"   {name} {s} {json.dumps(r['metrics'][s])}  | nemo {json.dumps(report['nemo'][s])}", flush=True)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cmd", choices=["prep", "cli", "ws", "compare"])
ap.add_argument("--work", required=True)
ap.add_argument("--vals", default="")
ap.add_argument("--per-set", type=int, default=200)
ap.add_argument("--post-pad-s", type=float, default=2.0)
ap.add_argument("--pack", default="")
ap.add_argument("--bin", default="./mynah-asr")
ap.add_argument("--jobs", type=int, default=48)
ap.add_argument("--port", type=int, default=8291)
ap.add_argument("--conc", type=int, default=16)
ap.add_argument("--realtime", type=int, default=0)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--nemo-dump", default="")
ap.add_argument("--runs", default="")
ap.add_argument("--lang", default="fr")
ap.add_argument("--out", default="")
a = ap.parse_args()
{"prep": prep, "cli": cli, "ws": ws, "compare": compare}[a.cmd](a)
