#!/bin/bash
# A2 = A0 with the VoxPopuli source filtered by an independent teacher, SAME weight (0.20).
# 1. Nemotron 0.6B in mynah-asr-server-cuda (GPU lock held), every VP train clip streamed
#    through it (teacher_ws.py): an annotated manifest with the teacher WER per utterance
#    (kept: other thresholds need no re-run). Gate first: the server transcripts of 300
#    clips must agree with the CLI teacher transcripts of the same clips.
# 2. filter teacher_wer <= ${THR:-50} (inclusive; None = no reference -> dropped) -> train_vp_t${THR}.json
# 3. mix_arm.sh it-a2-vpt${THR} with VP=<filtered>, nothing else changed (FILTER_ONLY=1 stops before it).
#   FT_ROOT=/root/ft tmux new -d -s a2 'FT_ROOT=/root/ft bash finetune/jobs/a2_vpfilter.sh 2>&1 | tee -a /root/ft/logs/a2.log'
# LANG_=fr (FR-P1): mix/fr, teacher_ws/teacher_check --lang fr (server lang = normaliser lang); the 300-clip CLI
# check file audit-fr/teacher_train_vp.json is built first if missing (Mynah CLI, CPU). LANG_=fr FILTER_ONLY=1 for FR-P1.
. "$(dirname "$0")/gpu_lock.sh"
THR=${THR:-50}; LANG_=${LANG_:-it}; MX=$FT_ROOT/mix/$LANG_; L=$FT_ROOT/logs
A=$FT_ROOT/audit-a0; [ "$LANG_" = it ] || A=$FT_ROOT/audit-$LANG_
cd "$FINETUNE/.." || exit 2
ulimit -n 65536
if [ ! -s $MX/train_vp.teacher.json ]; then
    if [ ! -s $A/teacher_train_vp.json ]; then   # CLI teacher on 300 seeded clips (IT: built by hand before A2)
        [ "$LANG_" = it ] && { echo "== missing $A/teacher_train_vp.json"; exit 3; }
        mkdir -p $A
        timeout 3600 python3 finetune/eou/diag/teacher_check.py --manifest $MX/train_vp.json --n 300 --lang $LANG_ \
            --jobs ${CHECK_JOBS:-24} --json $A/teacher_train_vp.json > $L/teacher_check_vp_$LANG_.log 2>&1 \
            || { tail -3 $L/teacher_check_vp_$LANG_.log; exit 3; }
        head -1 $L/teacher_check_vp_$LANG_.log | cut -c1-300
    fi
    gpu_lock
    ./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b -p 8295 > $L/teacher_srv.log 2>&1 & SP=$!
    for i in $(seq 120); do curl -sf localhost:8295/v1/health >/dev/null && break; sleep 1; done
    grep '^\[SERVER-CONFIG\]' $L/teacher_srv.log | cut -c1-200
    timeout 900 python3 finetune/eou/diag/teacher_ws.py --manifest $MX/train_vp.json --out $A/check_vp.teacher.json \
        --check $A/teacher_train_vp.json --lang $LANG_ --conc 64 || { kill $SP; exit 3; }
    timeout 3600 python3 finetune/eou/diag/teacher_ws.py --manifest $MX/train_vp.json --out $MX/train_vp.teacher.json \
        --lang $LANG_ --conc 128
    kill $SP; wait $SP 2>/dev/null
    flock -u 9
fi
python3 - $MX/train_vp.teacher.json $MX/train_vp_t$THR.json $THR <<'PY'
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]; thr = float(sys.argv[3])
keep = [r for r in rs if r["teacher_wer"] is not None and r["teacher_wer"] <= thr]   # inclusive; None -> dropped
with open(sys.argv[2], "w") as f:
    for r in keep:
        f.write(json.dumps({k: v for k, v in r.items() if k not in ("teacher", "teacher_wer")}, ensure_ascii=False) + "\n")
h = lambda x: sum(r["duration"] for r in x) / 3600  # noqa: E731
print(f"== VP filter teacher_wer <= {thr}: kept {len(keep)}/{len(rs)} utts, {h(keep):.2f}/{h(rs):.2f} h, "
      f"speakers {len({r['speaker'] for r in keep})}/{len({r['speaker'] for r in rs})}")
PY
[ -n "${FILTER_ONLY:-}" ] && exit 0   # resume_it.sh: the filtered manifest only, no arm
[ "$LANG_" = it ] || { echo "== no default arm for $LANG_ (FR-P1: fr_p1.sh)"; exit 0; }
VP=$MX/train_vp_t$THR.json bash "$FINETUNE/jobs/mix_arm.sh" it-a2-vpt$THR
echo "== A2-DONE $(date +%T)"
