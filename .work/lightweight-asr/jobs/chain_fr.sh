#!/bin/bash
# French replica of the Italian two-stage EOU path, NO tuning: same steps, lr,
# freeze policy, curriculum, best-selection, de-accented targets (stock SPE).
#   tmux new -d -s chainfr 'bash /root/chain_fr.sh 2>&1 | tee /root/ftfr/logs/chain.log'
set -u
PY=/root/nemo-venv/bin/python; M=/root/ftfr/manifests; L=/root/ftfr/logs; R=/root/ft/runs
until [ -s $M/train_40h.json ] && [ -s $M/eval_mls_fr.json ] && ! tmux has-session -t datafr 2>/dev/null; do sleep 20; done
echo "== chainfr data ready $(date +%T)"; wc -l $M/train_40h.json $M/eval_mls_fr.json
POL="--freeze-enc 1 --unfreeze-top 4 --enc-lr 3e-5"; VAL="--val $M/eval_mls_fr.json --val-n 200"
exec 9>/root/gpu.lock; flock 9
cd /root/eou-kit
$PY plain_it2.py --manifest $M/train_40h.json --steps 4000 --bs 16 --lr 3e-4 $POL --eval-every 500 $VAL \
    --tag fr-p40 2>&1 | grep --line-buffered -v -E "NeMo W|warn" > $L/plain-fr-p40.log
echo "== chainfr p40 done $(date +%T)"; grep VAL $L/plain-fr-p40.log | tail -2
$PY plain_it2.py --init $R/plain-fr-p40/final.nemo --manifest $M/train_40h.json --steps 1500 --bs 16 --lr 1e-4 $POL \
    --eou-append 1 --pad-prob 0.5 --pad-min 1 --pad-max 3 --fastemit 0 --eval-every 250 $VAL \
    --tag fr-s2-eou 2>&1 | grep --line-buffered -v -E "NeMo W|warn" > $L/plain-fr-s2.log
echo "== chainfr s2 done $(date +%T)"; grep VAL $L/plain-fr-s2.log | tail -2
flock -u 9
# Mynah: export + stream WER on FLEURS-fr (samples/eval-bank, fr) + eou_metrics (40 clips, gap 1 s)
P=/root/mynah-asr/models/parakeet-realtime-eou-120m-fr
MYNAH=/root/mynah-asr PY=$PY timeout 1500 bash /root/eou-kit/export_to_mynah.sh $R/plain-fr-s2-eou/final.nemo $P 2>&1 | tail -2
cd /root/mynah-eou
$PY - <<'PYEOF'
import json, re, subprocess, unicodedata, concurrent.futures as cf
man = [s for s in json.load(open("samples/eval-bank/manifest.json"))["samples"] if s.get("lang") == "fr"]
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
    out = subprocess.run(["./mynah-asr", "stream", "-m", "models/parakeet-realtime-eou-120m-fr", "-i", "samples/eval-bank/" + s["file"],
                          "--quant", "f32", "--deltas"], capture_output=True, text=True).stdout
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
print(f"MYNAH-FR n={len(res)} WER {100*we/wn:.2f} CER {100*ce/cn:.2f} (accent-insensitive, pooled) empty {emp} clips_with_model_eou {eo}")
PYEOF
timeout 1500 python3 tools/eval/eou_metrics.py -m models/parakeet-realtime-eou-120m-fr --manifest samples/eval-bank/manifest.json \
    --root samples/eval-bank --langs fr --mode both --gaps 1 --vad models/silero-vad --limit 40 --jobs 14 \
    --work-dir /root/ftfr/eouw --json /root/ftfr/eou-metrics.json > /root/ftfr/eou-metrics.txt 2>&1; echo "rc=$?"
grep -E "speech end ->|missed|premature \(any\)" /root/ftfr/eou-metrics.txt; tail -3 /root/ftfr/eou-metrics.txt
echo "== CHAINFR-DONE $(date +%T)"
