#!/bin/bash
# S15-21 P2, first arm: the LABEL SOURCE alone. E3 recipe (stock -> whole encoder 2e-4 from step 0,
# decoder+joint 3e-4, A2 mix, seed 0, 4000 steps), the SAME clips in the SAME order (so the seeded
# sampler draws the same audio batches), only the transcript changes: human reference -> Nemotron
# 0.6B teacher transcript (numbers verbalised by eou/verbalize_it.py, not dropped). A clip whose
# teacher transcript is empty, or whose number reading is a guess (phone, 802.11, 24/7...), keeps
# its human label (counted; the arm must stay batch-paired).
# 1. teacher_ws.py over the A2 sources that have no teacher scores yet (MLS 40 h, CV, FLEURS; VP is
#    archived: train_vp.teacher.json), GPU lock held, server killed afterwards;
# 2. mix/it-tl/*.json: text = verbalised teacher, ref_human = the original text; per-source stats
#    (teacher vs human WER, digits verbalised, fallbacks);
# 3. mix_arm.sh it-p2-tl (LABELS_ONLY=1 stops before it), then the CPU greedy/beam-8/oracle probe.
# Compare with E3 = plain-it-e2-enc2e-4-s4000 (36.52 / beam 33.09 / oracle 28.52): val refs stay human.
#   FT_ROOT=/root/ft tmux new -d -s p2 'FT_ROOT=/root/ft bash finetune/jobs/p2_labels.sh 2>&1 | tee -a /root/ft/logs/p2_labels.log'
. "$(dirname "$0")/gpu_lock.sh"
MF=$FT_ROOT/manifests; MX=$FT_ROOT/mix/it; TL=$FT_ROOT/mix/it-tl; L=$FT_ROOT/logs; R=$FT_ROOT/runs
mkdir -p $TL; cd "$FINETUNE/.." || exit 2; ulimit -n 65536
echo "== p2_labels $(date +%T)"
declare -A SRC=([mls]=$MF/train_40h.json [cv]=$MX/train_cv.json [fleurs]=$MX/train_fleurs.json)
todo=; for s in mls cv fleurs; do [ -s $TL/$s.teacher.json ] || todo="$todo $s"; done
if [ -n "$todo" ]; then
    gpu_lock
    ./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv_tl.log 2>&1 & SP=$!
    for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
    for s in $todo; do
        timeout 3600 python3 finetune/eou/diag/teacher_ws.py --manifest ${SRC[$s]} --out $TL/$s.teacher.json \
            --lang it --conc 128 | tail -1 | cut -c1-200
    done
    kill $SP; wait $SP 2>/dev/null; flock -u 9
fi
python3 - $TL $MX/train_vp.teacher.json $MX/train_vp_t50.json <<'PY' || exit 3
import json, sys
sys.path.insert(0, "finetune/eou"); from verbalize_it import verbalize_ex
GUESS = {"phone", "longdigits", "dotnum", "slash", "date2y", "residual"}   # verbalize_it.py --drop-flags default
tl, vp_scores, vp_t50 = sys.argv[1:]
by_path = {json.loads(l)["audio_filepath"]: json.loads(l) for l in open(vp_scores)}
srcs = {s: [json.loads(l) for l in open(f"{tl}/{s}.teacher.json")] for s in ("mls", "cv", "fleurs")}
srcs["vp"] = [by_path[json.loads(l)["audio_filepath"]] for l in open(vp_t50)]
for s, rs in srcs.items():
    fb = dig = guess = 0
    with open(f"{tl}/{s}.json", "w") as f:
        for r in rs:
            raw = (r.get("teacher_raw") or "").strip()
            row = {k: v for k, v in r.items() if k not in ("teacher", "teacher_raw", "teacher_lang", "teacher_wer")}
            row["ref_human"] = r["text"]; row["teacher_wer"] = r.get("teacher_wer")
            vb, flags = verbalize_ex(raw)
            if not raw:
                fb += 1
            elif GUESS & set(flags):   # a guessed reading: keep the clip (batch pairing), human label
                guess += 1
            else:
                dig += any(c.isdigit() for c in raw)
                row["text"] = vb
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    w = sorted(r["teacher_wer"] for r in rs if r.get("teacher_wer") is not None)
    print(f"== TL {s}: {len(rs)} clips {sum(r['duration'] for r in rs) / 3600:.1f} h; teacher vs human WER "
          f"p50 {w[len(w) // 2]:.1f} p90 {w[int(len(w) * 0.9)]:.1f} mean {sum(w) / len(w):.1f}; "
          f"digits verbalised {dig}; human label kept: empty {fb}, guessed number reading {guess}")
PY
[ -n "${LABELS_ONLY:-}" ] && exit 0
TLMIX="$TL/mls.json:0.35,$TL/cv.json:0.30,$TL/vp.json:0.20,$TL/fleurs.json:0.15"
MIX="$TLMIX" POL="--freeze-enc 0 --enc-lr 2e-4 --lr 3e-4 --bs 16" SMOKE=1 bash "$FINETUNE/jobs/mix_arm.sh" it-p2-tl \
    --eval-steps 100,250 --abort-wer 1000:85
[ -s $R/plain-it-p2-tl/final.nemo ] && CUDA_VISIBLE_DEVICES= timeout 3600 "$PY" "$FINETUNE/eou/diag/lm_fusion_probe.py" \
    --nemo $R/plain-it-p2-tl/final.nemo --per-set 200 --alphas 0 --beam 8 --beam-alphas 0 \
    --out $R/plain-it-p2-tl/beam_probe.json 2>&1 | grep -E "macro|Traceback|Error" | tail -4
cd $FT_ROOT && python3 "$FINETUNE/eou/val_table.py" --col E3=runs/plain-it-e2-enc2e-4-s4000/metrics.json:4000 \
    --col TL=runs/plain-it-p2-tl/metrics.json:4000 --delta TL-E3 --trajectory runs/plain-it-p2-tl/metrics.json
echo "== P2-TL-DONE $(date +%T)"
