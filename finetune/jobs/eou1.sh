#!/bin/bash
# eou1: Italian-specialist FT of parakeet_realtime_eou_120m-v1 (finetune/eou/).
# EXPERIMENTAL / KNOWN-BAD: with LR=1e-3 EOU_WEIGHT=0.9 (the 2026-10-09 values,
# new Italian SPE, FastEmit 0.03) every arm collapsed to blank. LR and
# EOU_WEIGHT are therefore required; this job exists to reproduce and ablate. Download + tokenizer run OUTSIDE the GPU lock;
# training, evals, streaming check, Mynah export + stream sanity under it.
# Reuses the Canary kit's data (/root/ft/manifests, /root/ft/text/it_pool.txt).
# Env: FT_ROOT (required), LR and EOU_WEIGHT (required), SUBSET (5h), EPOCHS (52:
#      ~260 audio-h of unpadded speech on 5 h, like the Canary curve), ENC_LR_SCALE (0.3),
#      FREEZE (0), WARM_DEC_JOINT (0), BATCH_DURATION (300), RATE_USD_H, T_TRAIN (10800 s),
#      MYNAH (/root/mynah-asr). Needs setup.sh with WITH_RNNT=1 (numba-cuda, numpy<2.4).
#   FT_ROOT=/root/ft LR=1e-3 EOU_WEIGHT=0.9 tmux new -d -s eou1 'bash finetune/jobs/eou1.sh 2>&1 | tee -a /root/ft/logs/eou1.log'
set -u
. "$(dirname "$0")/gpu_lock.sh"
KIT=$FINETUNE/eou
FT=$FT_ROOT
: "${LR:?set LR explicitly (1e-3 = the known-bad 2026-10-09 value)}"
: "${EOU_WEIGHT:?set EOU_WEIGHT explicitly (0.9 = the known-bad 2026-10-09 value)}"
MYNAH=${MYNAH:-/root/mynah-asr}
SUBSET=${SUBSET:-5h}; EPOCHS=${EPOCHS:-52}
TAG=${TAG:-eou-it-$SUBSET-e$EPOCHS}
STOCK=$FT/models/parakeet_realtime_eou_120m-v1.nemo
TOK=$FT/models/tok-eou-it
URL=https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/resolve/main/parakeet_realtime_eou_120m-v1.nemo
mkdir -p "$FT/models" "$FT/logs"
die() { echo "== EOU1-FAILED $1 rc=$2 $(date +%T)"; exit 1; }
for f in ../common/ftlib.py tokenizer_eou.py train_eou.py export_to_mynah.sh; do
    [ -e "$KIT/$f" ] || die "missing $KIT/$f (copy the whole finetune/ directory)" 2
done
[ -s "$FT/manifests/train_$SUBSET.json" ] && [ -s "$FT/text/it_pool.txt" ] || die "Canary-kit data missing under $FT" 2
echo "== eou1 start $(date +%T) free $(df -h --output=avail "$FT" | tail -1)"

# 1. stock .nemo, resumable (public, no token)
if [ ! -s "$STOCK" ]; then
    timeout 1800 curl -fL --retry 5 --retry-delay 5 -C - -o "$STOCK.part" "$URL" || die download $?
    tar -tf "$STOCK.part" >/dev/null || die "download not a tar" 1
    mv "$STOCK.part" "$STOCK"
fi
ls -l "$STOCK"

# 2. Italian SPE + <EOU>/<EOB> (CPU)
if [ ! -s "$TOK/tokenizer_info.json" ]; then
    timeout 1800 "$PY" "$KIT/tokenizer_eou.py" --stock "$STOCK" --text "$FT/text/it_pool.txt" --out "$TOK" \
        2>&1 | tail -20
    [ "${PIPESTATUS[0]}" = 0 ] || die tokenizer "${PIPESTATUS[0]}"
fi

# 3-5 under the GPU lock
gpu_lock
echo "== eou1 lock held $(date +%T) tag=$TAG"
f=$(df -BG --output=avail "$FT" | tail -1 | tr -dc 0-9)
[ "${f:-0}" -ge 3 ] || die "disk: ${f} GB free, need 3 (final.nemo 0.46 + last.ckpt ~1.4 + pack 0.46)" 4
RUN=$FT/runs/$TAG
if [ ! -s "$RUN/metrics.json" ]; then
    timeout "${T_TRAIN:-10800}" "$PY" "$KIT/train_eou.py" --stock "$STOCK" --tokenizer "$TOK" --subset "$SUBSET" \
        --epochs "$EPOCHS" --lr "$LR" --eou-weight "$EOU_WEIGHT" --warm-dec-joint "${WARM_DEC_JOINT:-0}" --enc-lr-scale "${ENC_LR_SCALE:-0.3}" \
        --freeze-encoder-layers "${FREEZE:-0}" --batch-duration "${BATCH_DURATION:-300}" \
        --save-ckpt 1 --rate-usd-h "${RATE_USD_H:-0}" --tag "$TAG" 2>&1 \
        | tee "$FT/logs/train-$TAG.log" | grep --line-buffered -E "^(==|  step|  VAL|  \[|  STREAM|  model|  ckpt|Traceback|[A-Za-z]*Error)"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || die train "$rc"
fi
MYNAH=$MYNAH PY=$PY timeout 1800 bash "$KIT/export_to_mynah.sh" "$RUN/final.nemo" "$MYNAH/models/parakeet-realtime-eou-120m-it" \
    2>&1 | tee "$FT/logs/export-$TAG.log"

# summary
"$PY" - "$RUN/metrics.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
f = m["final_it"]
print("== EOU1 SUMMARY", m["tag"])
for k in ("fleurs_it", "mls_it"):
    print(f"  {k}: WER {f[k]['wer']} CER {f[k]['cer']} empty {f[k]['empty_hyp']} eou_rate {f[k]['eou_emitted_rate']}")
print(f"  fleurs_en (info): WER {m['en_after']['fleurs_en']['wer']}")
for lv, r in m["level_sweep"].items():
    print(f"  @{lv} dBFS: " + " ".join(f"{k} {v['wer']}/{v['empty_hyp']}e/{v['eou_emitted_rate']}eou" for k, v in {**r['it'], **r['en']}.items()))
s = m["streaming_check"]
print(f"  streaming (NeMo cache-aware): n_ok {s['n_ok']} eou_rate {s['eou_rate']} text_match {s['text_match_rate']}")
print("  " + json.dumps({k: m[k] for k in ("steps_done", "audio_hours_processed", "audio_h_per_gpu_h", "samples_per_s",
      "peak_alloc_gb", "peak_reserved_gb", "train_wall_s", "gpu_hours_total", "cost_usd_total")}))
print("  ckpt", m["checkpoint"])
PYEOF
df -h --output=avail "$FT" | tail -1
echo "== EOU1-DONE $(date +%T)"
