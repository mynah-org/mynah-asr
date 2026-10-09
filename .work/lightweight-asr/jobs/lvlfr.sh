#!/bin/bash
# FLEURS-fr level check for the EOU-FR pack: same 200 clips, peak-normalised to
# -3 dBFS (diagnostic transform, as lvl.sh did for the English stock), stream WER.
set -u
until grep -q CHAINFR-DONE /root/ftfr/logs/chain.log 2>/dev/null; do sleep 15; done
cd /root/mynah-eou
/root/nemo-venv/bin/python - <<'PY'
import json, os, re, subprocess, unicodedata, wave, numpy as np, concurrent.futures as cf
src = "samples/eval-bank"; dst = "/root/ftfr/bank-fr-peak"
man = [s for s in json.load(open(f"{src}/manifest.json"))["samples"] if s.get("lang") == "fr"]
def rd(p):
    w = wave.open(p); return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
peaks = []
for s in man:
    x = rd(f"{src}/{s['file']}"); pk = np.abs(x).max() + 1e-9; peaks.append(20 * np.log10(pk))
    y = x * (10 ** (-3 / 20) / pk); os.makedirs(os.path.dirname(f"{dst}/{s['file']}"), exist_ok=True)
    w = wave.open(f"{dst}/{s['file']}", "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
print(f"FR bank peak dBFS p10/p50/p90 {np.percentile(peaks,10):.1f}/{np.percentile(peaks,50):.1f}/{np.percentile(peaks,90):.1f}")
def norm(t):
    t = unicodedata.normalize("NFKD", t.lower()); t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]
def run(s):
    out = subprocess.run(["./mynah-asr", "stream", "-m", "models/parakeet-realtime-eou-120m-fr", "-i", f"{dst}/{s['file']}",
                          "--quant", "f32", "--deltas"], capture_output=True, text=True).stdout
    t, e = "", 0
    for l in out.splitlines():
        if l.startswith("{"):
            o = json.loads(l)
            if o.get("type") == "final": t = o.get("text", "")
            if o.get("type") == "eou" and o.get("source") == "model": e += 1
    return s, t, e
with cf.ThreadPoolExecutor(14) as ex: res = list(ex.map(run, man))
we = wn = ce = cn = emp = eo = 0
for s, h, e in res:
    r, h = norm(s["text"]), norm(h); we += lev(r.split(), h.split()); wn += len(r.split()); ce += lev(r, h); cn += len(r); emp += not h; eo += e > 0
print(f"MYNAH-FR-PEAK n={len(res)} WER {100*we/wn:.2f} CER {100*ce/cn:.2f} empty {emp} clips_with_model_eou {eo}")
PY
echo "== LVLFR-DONE $(date +%T)"
