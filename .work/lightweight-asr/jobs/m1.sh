#!/bin/bash
# m1: the stage-2 EOU-IT best checkpoint inside Mynah.
#  1. export to a Mynah pack (convert_nemo.py)
#  2. CLI stream on FLEURS-it test (200, samples/eval-bank-it): accent-insensitive
#     WER/CER (the stock tokenizer has no accented vowels), empty rate, EOU rate
#  3. eou_metrics (speech end -> EOU, premature, missed, A + 1 s + B)
#  4. CUDA engine parity + serving-loop gates (tests/test_cuda_stream) on IT clips
#  5. CUDA server start-up line (VRAM)
#   NEMO=/root/ft/runs/plain-s2-eou-.../final.nemo tmux new -d -s m1 'bash /root/m1.sh 2>&1 | tee /root/ft/logs/m1.log'
set -u
N=/root/mynah-eou; PY=/root/nemo-venv/bin/python; O=/root/ft/m1; mkdir -p $O
NEMO=${NEMO:?}; PACK=$N/models/parakeet-realtime-eou-120m-it
echo "== m1 export $(date +%T) $NEMO"
MYNAH=/root/mynah-asr PY=$PY CLIPS="$(ls $N/samples/eval-bank-it/it/*.wav | head -3)" \
    timeout 1500 bash /root/eou-kit/export_to_mynah.sh "$NEMO" /root/mynah-asr/models/parakeet-realtime-eou-120m-it 2>&1 | tail -6
cd $N || exit 2
echo "== m1 stream WER $(date +%T)"
$PY - <<'PY'
import json, re, subprocess, unicodedata, concurrent.futures as cf
man = json.load(open("samples/eval-bank-it/manifest.json"))["samples"]
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
    out = subprocess.run(["./mynah-asr", "stream", "-m", "models/parakeet-realtime-eou-120m-it", "-i",
                          "samples/eval-bank-it/" + s["file"], "--quant", "f32", "--deltas"], capture_output=True, text=True).stdout
    text, eous = "", 0
    for l in out.splitlines():
        if l.startswith("{"):
            o = json.loads(l)
            if o.get("type") == "final": text = o.get("text", "")
            if o.get("type") == "eou" and o.get("source") == "model": eous += 1
    return s, text, eous
with cf.ThreadPoolExecutor(14) as ex: res = list(ex.map(run, man))
we = wn = ce = cn = emp = eo = 0
for s, h, e in res:
    r, h = norm(s["text"]), norm(h)
    we += lev(r.split(), h.split()); wn += len(r.split()); ce += lev(r, h); cn += len(r); emp += not h; eo += e > 0
print(f"MYNAH-IT n={len(res)} WER {100*we/wn:.2f} CER {100*ce/cn:.2f} (accent-insensitive, pooled) empty {emp} clips_with_model_eou {eo}")
for s, h, e in res[:3]: print("  ref:", norm(s["text"])[:80], "\n  hyp:", norm(h)[:80], "| eous", e)
PY
echo "== m1 eou_metrics $(date +%T)"
timeout 1500 python3 tools/eval/eou_metrics.py -m models/parakeet-realtime-eou-120m-it --manifest samples/eval-bank-it/manifest.json \
    --root samples/eval-bank-it --langs it --mode both --gaps 1 --vad models/silero-vad --limit 60 --jobs 14 \
    --work-dir $O/eouw --json $O/eou-metrics.json > $O/eou-metrics.txt 2>&1; echo "rc=$?"; tail -12 $O/eou-metrics.txt
echo "== m1 CUDA parity $(date +%T)"
exec 9>/root/gpu.lock; flock 9
timeout 900 tests/test_cuda_stream models/parakeet-realtime-eou-120m-it --lookahead 1 $(ls samples/eval-bank-it/it/*.wav | head -4) > $O/ts.txt 2>&1
echo "rc=$?"; grep -E "^(OK|FAIL|PASS)" $O/ts.txt | grep -E "B cpu|PASS|FAIL" | head -8
./mynah-asr-server-cuda -m models/parakeet-realtime-eou-120m-it -p 8291 > $O/srv.log 2>&1 & P=$!
for i in $(seq 60); do curl -sf localhost:8291/v1/health >/dev/null && break; sleep 1; done
grep '^\[SERVER-CONFIG\]' $O/srv.log | cut -c1-260; kill $P; wait $P 2>/dev/null
echo "== M1-DONE $(date +%T)"
