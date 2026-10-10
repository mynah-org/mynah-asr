#!/bin/bash
# A0 failure audit on VoxPopuli / Common Voice (no training): the same val clips decoded by
# P40 (MLS-only, 2026-10-09) and A0 (multi-domain), overlap table, every A0 failure with
# ref / hyp / duration / level / speaker / id; then an independent teacher (Nemotron 0.6B
# in Mynah, CPU) on the failed clips: if the teacher reads the reference, the speech IS in
# the segment and the failure is the student's. Failed wavs are kept for listening.
#   FT_ROOT=/root/ft tmux new -d -s audit 'FT_ROOT=/root/ft bash finetune/jobs/a0_audit.sh 2>&1 | tee /root/ft/logs/a0_audit.log'
. "$(dirname "$0")/gpu_lock.sh"
O=$FT_ROOT/audit-a0; rm -rf $O; mkdir -p $O
gpu_lock
timeout 1200 "$PY" "$FINETUNE/eou/diag/empty_clips.py" --nemo $FT_ROOT/runs/plain-it-a0-mix/final.nemo \
    --ref-nemo $FT_ROOT/base/plain-p40-b2pol/final.nemo --dump $O \
    --val vp=$FT_ROOT/mix/it/eval_vp.json,cv=$FT_ROOT/mix/it/eval_cv.json 2>&1 | grep -E "^==|^   "
flock -u 9
echo "== teacher (Nemotron 0.6B, Mynah CPU, --lang it) on the failures $(date +%T)"
cd "$FINETUNE/.." && "$PY" - "$O" <<'PY'
import json, re, subprocess, sys, unicodedata, concurrent.futures as cf
from pathlib import Path
O = Path(sys.argv[1]); F = json.load(open(O / "failures.json"))
def norm(t):
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z' ]", " ", t).split())
def lev(x, y):
    p = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        c = [i]
        for j, v in enumerate(y, 1): c.append(min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (u != v)))
        p = c
    return p[-1]
def run(x):
    o = subprocess.run(["./mynah-asr", "transcribe", "-m", "models/nemotron-3.5-asr-streaming-0.6b", "-i", x["audio_filepath"],
                        "--lang", "it"], capture_output=True, text=True, timeout=300)
    return x, o.stdout.strip().splitlines()[-1] if o.stdout.strip() else "", o.stderr[-200:]
with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(run, F))
print("raw teacher output of the first clip:", repr(res[0][1][:160]), res[0][2][-120:].replace("\n", " ") if not res[0][1] else "")
for x, t, _ in res:
    tw = 100 * lev(x["ref"].split(), norm(t).split()) / max(1, len(x["ref"].split()))
    x["teacher"], x["teacher_wer"] = t, round(tw, 1)
    print(f"  {x['set']:2s} {Path(x['audio_filepath']).stem[:40]:40s} {x['duration']:5.1f}s rms {x['rms_dbfs']:6.1f} | teacher WER {tw:5.1f}% | P40 '{(x['ref_model_hyp'] or '')[:40]}' | A0 '{x['hyp'][:30]}'")
json.dump(F, open(O / "failures.json", "w"), indent=1, ensure_ascii=False)
for s in ("vp", "cv"):
    v = [x["teacher_wer"] for x in F if x["set"] == s]
    if v:
        print(f"== {s}: {len(v)} A0 failures, teacher WER median {sorted(v)[len(v)//2]:.1f}%, teacher <= 30 % on {sum(w <= 30 for w in v)}/{len(v)}")
PY
echo "== A0-AUDIT-DONE $(date +%T)"
