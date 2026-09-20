#!/usr/bin/env python3
"""rest_load.py — offline (REST) concurrency for models that cannot stream.

The streaming harness (stream_load.py) answers "how many real-time streams does
this box hold". That question does not exist for a model the runtime refuses to
stream: `mynah_asr_stream_unsupported()` turns away every pack with linear
biases, a folded batch_norm or symmetric conv padding, which is every Parakeet
pack converted so far. For those the concurrency question is a THROUGHPUT
question — how many files at once before latency stops being worth it — and it
is measured on POST /v1/audio/transcriptions.

Two numbers per rung, and they mean different things:

  * xRT      = (audio seconds accepted) / (wall seconds of the rung).
               The fleet's throughput. It is the number that pays for hardware.
  * latency  = per request, submit to final JSON. It is what a caller waits.

Throughput rises with concurrency until the cores are full and then flattens
while latency keeps climbing; the useful rung is the last one before that knee.
This tool reports both and names the knee, it does not choose for you.

A refusal (503) is data, not an error: the server saying it is full is the
admission ladder working. Errors that are NOT 503 invalidate the rung.

Usage:
    python3 tools/bench/rest_load.py --port 8090 --clips a.wav b.wav \\
        --ladder 1,2,4,8,16,32 --requests-per-stream 2 --lang en --json out.json
"""
import argparse, json, mimetypes, os, queue, ssl, sys, threading, time, urllib.error, urllib.request, wave

def wav_seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())

def multipart(fields, fname, blob):
    """Build one multipart body. Boundary is fixed per call, not global: two
    threads building bodies at once must not share it."""
    b = "----mynah%016x" % int.from_bytes(os.urandom(8), "big")
    out = bytearray()
    for k, v in fields.items():
        out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    out += (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{os.path.basename(fname)}\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode()
    out += blob + f"\r\n--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"

def one(url, fields, path, blob, timeout):
    body, ctype = multipart(fields, path, blob)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            txt = json.loads(r.read().decode("utf-8", "replace")).get("text", "")
        return {"ok": True, "ms": (time.monotonic() - t0) * 1e3, "text": txt}
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")[:200]
        return {"ok": False, "ms": (time.monotonic() - t0) * 1e3,
                "refused": e.code == 503, "code": e.code, "error": payload}
    except Exception as e:                                   # connect, timeout, reset
        return {"ok": False, "ms": (time.monotonic() - t0) * 1e3,
                "refused": False, "code": 0, "error": f"{type(e).__name__}: {e}"}

def pct(xs, p):
    if not xs: return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]

def rung(a, clips, blobs, secs, c):
    """One concurrency rung: c threads, each sending --requests-per-stream
    requests back to back. Wall is measured across the whole rung, so xRT is
    the fleet's rate and not the sum of per-request rates."""
    url = f"http://{a.host}:{a.port}/v1/audio/transcriptions"
    fields = {"language": a.lang, "response_format": "json"}
    if a.model: fields["model"] = a.model
    res, lock = [], threading.Lock()
    def worker(w):
        for k in range(a.requests_per_stream):
            i = (w + k) % len(clips)
            r = one(url, fields, clips[i], blobs[i], a.timeout)
            r["audio_s"] = secs[i]
            with lock: res.append(r)
    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(c)]
    t0 = time.monotonic()
    for t in ths: t.start()
    for t in ths: t.join(timeout=max(30.0, a.timeout * a.requests_per_stream + 30))
    wall = time.monotonic() - t0
    ok = [r for r in res if r["ok"]]
    refused = [r for r in res if not r["ok"] and r.get("refused")]
    broke = [r for r in res if not r["ok"] and not r.get("refused")]
    lat = [r["ms"] for r in ok]
    audio = sum(r["audio_s"] for r in ok)
    return {"concurrency": c, "wall_s": round(wall, 2), "sent": len(res), "ok": len(ok),
            "refused": len(refused), "errors": len(broke),
            "error_sample": broke[0]["error"] if broke else None,
            "audio_s": round(audio, 1), "xrt": round(audio / wall, 2) if wall > 0 else None,
            "lat_p50_ms": round(pct(lat, 50) or 0, 1), "lat_p95_ms": round(pct(lat, 95) or 0, 1),
            "lat_max_ms": round(max(lat), 1) if lat else None,
            "texts": [r["text"] for r in ok[:2]]}

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--clips", nargs="+", required=True, help="16 kHz mono s16 WAVs")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--model", default=None, help="worker group, in a multi-model fleet")
    ap.add_argument("--ladder", default="1,2,4,8,16,32", help="concurrency rungs")
    ap.add_argument("--requests-per-stream", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json", help="write every rung here")
    a = ap.parse_args()

    clips = list(a.clips)
    blobs = [open(c, "rb").read() for c in clips]
    secs = [wav_seconds(c) for c in clips]
    ladder = [int(x) for x in a.ladder.split(",") if x.strip()]

    print("rest_load — offline concurrency (throughput, not real-time streams)")
    print(f"  server http://{a.host}:{a.port}  clips {len(clips)}  "
          f"audio {sum(secs):.1f} s  lang={a.lang}  model={a.model or 'default'}")
    print(f"  {'C':>4} {'ok':>5} {'ref':>4} {'err':>4} {'wall_s':>7} "
          f"{'xRT':>7} {'p50_ms':>8} {'p95_ms':>8}")
    rows = []
    for c in ladder:
        r = rung(a, clips, blobs, secs, c)
        rows.append(r)
        print(f"  {r['concurrency']:>4} {r['ok']:>5} {r['refused']:>4} {r['errors']:>4} "
              f"{r['wall_s']:>7.2f} {str(r['xrt']):>7} {r['lat_p50_ms']:>8.1f} "
              f"{r['lat_p95_ms']:>8.1f}"
              + ("   <- ERRORS, rung invalid" if r["errors"] else ""))
        if r["errors"]:
            print(f"       {r['error_sample']}")

    # The knee: the last rung whose xRT still gained more than 5 % on the one
    # before it. Named, not chosen — a caller with a latency budget may want an
    # earlier rung, and this tool does not know that budget.
    best, knee = 0.0, None
    for r in rows:
        x = r["xrt"] or 0.0
        if x > best * 1.05:
            best, knee = x, r["concurrency"]
    if knee is not None:
        print(f"  knee: xRT stops gaining above C={knee} (peak {best} xRT)")
    print("  NOTE offline throughput. It is not a count of real-time streams: "
          "this model does not stream.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tool": "rest_load", "v": 1, "host": a.host, "port": a.port,
                       "lang": a.lang, "model": a.model, "clips": clips,
                       "requests_per_stream": a.requests_per_stream,
                       "knee_concurrency": knee, "peak_xrt": best, "rungs": rows}, f, indent=1)
        print(f"  -> {a.json}")
    return 0 if all(not r["errors"] for r in rows) else 1

if __name__ == "__main__":
    sys.exit(main())
