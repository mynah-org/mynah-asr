#!/bin/bash
# A2 = A0 with the VoxPopuli source filtered by an independent teacher, SAME weight (0.20).
# 1. Nemotron 0.6B in mynah-asr-server-cuda (GPU lock held), every VP train clip streamed
#    through it (teacher_ws.py): an annotated manifest with the teacher WER per utterance
#    (kept: other thresholds need no re-run). Gate first: the server transcripts of 300
#    clips must agree with the CLI teacher transcripts of the same clips.
# 2. filter teacher_wer <= ${THR:-50} -> train_vp_t${THR}.json
# 3. mix_arm.sh it-a2-vpt${THR} with VP=<filtered>, nothing else changed.
#   FT_ROOT=/root/ft tmux new -d -s a2 'FT_ROOT=/root/ft bash finetune/jobs/a2_vpfilter.sh 2>&1 | tee -a /root/ft/logs/a2.log'
. "$(dirname "$0")/gpu_lock.sh"
THR=${THR:-50}; MX=$FT_ROOT/mix/it; L=$FT_ROOT/logs; A=$FT_ROOT/audit-a0
cd "$FINETUNE/.." || exit 2
ulimit -n 65536
if [ ! -s $MX/train_vp.teacher.json ]; then
    gpu_lock
    ./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv.log 2>&1 & SP=$!
    for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
    grep '^\[SERVER-CONFIG\]' $L/teacher_srv.log | cut -c1-200
    timeout 900 python3 finetune/eou/diag/teacher_ws.py --manifest $MX/train_vp.json --out $A/check_vp.teacher.json \
        --check $A/teacher_train_vp.json --conc 64 || { kill $SP; exit 3; }
    timeout 3600 python3 finetune/eou/diag/teacher_ws.py --manifest $MX/train_vp.json --out $MX/train_vp.teacher.json --conc 128
    kill $SP; wait $SP 2>/dev/null
    flock -u 9
fi
python3 - $MX/train_vp.teacher.json $MX/train_vp_t$THR.json $THR <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]; thr = float(sys.argv[3])
keep = [r for r in rs if r["teacher_wer"] <= thr]
with open(sys.argv[2], "w") as f:
    for r in keep:
        f.write(json.dumps({k: v for k, v in r.items() if k not in ("teacher", "teacher_wer")}, ensure_ascii=False) + "\n")
h = lambda x: sum(r["duration"] for r in x) / 3600  # noqa: E731
print(f"== VP filter teacher_wer <= {thr}: kept {len(keep)}/{len(rs)} utts, {h(keep):.2f}/{h(rs):.2f} h, "
      f"speakers {len({r['speaker'] for r in keep})}/{len({r['speaker'] for r in rs})}")
PY
VP=$MX/train_vp_t$THR.json bash "$FINETUNE/jobs/mix_arm.sh" it-a2-vpt$THR
echo "== A2-DONE $(date +%T)"
