#!/bin/bash
# FT'd .nemo -> Mynah pack + Mynah-side streaming sanity (3 Italian clips).
#   bash finetune/eou/export_to_mynah.sh <final.nemo> [pack_dir]
# Env: MYNAH (repo with a built ./mynah-asr, default /root/mynah-asr),
#      PY (python with torch, yaml, numpy, safetensors: default /root/nemo-venv/bin/python,
#      since tools/.venv is gone on the box), CLIPS (wav list; default: the padded
#      streaming-check wavs the training run wrote next to final.nemo).
# convert_nemo.py takes a directory that holds ONLY the .nemo (no config.json):
# the .nemo is hard-linked in (no extra disk), converted (model.safetensors f32
# ~0.46 GB, mynah.json, tokens.json), then the link is removed.
set -u
NEMO=$(readlink -f "${1:?usage: export_to_mynah.sh <final.nemo> [pack_dir]}")
MYNAH=${MYNAH:-/root/mynah-asr}
PY=${PY:-/root/nemo-venv/bin/python}
PACK=${2:-$MYNAH/models/parakeet-realtime-eou-120m-it}
RUN=$(dirname "$NEMO")
mkdir -p "$PACK"
rm -f "$PACK"/*.nemo "$PACK/config.json"
ln "$NEMO" "$PACK/$(basename "$NEMO")" 2>/dev/null || cp "$NEMO" "$PACK/"
timeout 1200 "$PY" "$MYNAH/tools/convert_nemo.py" "$PACK" || { echo "== EXPORT-FAILED convert rc=$?"; exit 1; }
rm -f "$PACK"/*.nemo
"$PY" - "$PACK" <<'PYEOF'
import json, sys
p = sys.argv[1]
m = json.load(open(f"{p}/mynah.json")); t = json.load(open(f"{p}/tokens.json"))
d = m["decoder"]
print("  pack:", m["arch"], m["engine"], "vocab", d["vocab_size"], "blank", d["blank_id"],
      "presets", m["streaming"]["att_context_presets"], "normalize", m["features"]["normalize"])
assert t[1024] == "<EOU>" and t[1025] == "<EOB>" and d["blank_id"] == 1026 and len(t) == 1027, (t[1020:], d)
assert m["streaming"]["att_context_presets"][0] == [70, 1], m["streaming"]
print("  tokens: <EOU>=1024 <EOB>=1025 blank=1026 OK")
PYEOF
[ $? = 0 ] || { echo "== EXPORT-FAILED pack check"; exit 1; }
du -sh "$PACK"

# Mynah streaming sanity: transcript + eou events on 3 clips
BIN=$MYNAH/mynah-asr
[ -x "$BIN" ] || { echo "== no $BIN (build it: cd $MYNAH && make) -- skipping the stream sanity"; exit 0; }
CLIPS=${CLIPS:-$(ls "$RUN"/stream_wavs/*.wav 2>/dev/null | head -3)}
n=0; n_eou=0
for w in $CLIPS; do
    o=$(cd "$MYNAH" && timeout 300 "$BIN" stream -m "$PACK" -i "$w" --deltas 2>/dev/null)
    e=$(printf '%s\n' "$o" | grep -c '"eou"')
    txt=$(printf '%s\n' "$o" | grep -v '^{' | tail -1)
    echo "  MYNAH $(basename "$w"): eou_events=$e text=$(printf '%s' "$txt" | cut -c1-200)"
    printf '%s\n' "$o" | grep '^{' | tail -3 | cut -c1-200
    n=$((n + 1)); [ "$e" -gt 0 ] && n_eou=$((n_eou + 1))
done
echo "== MYNAH-STREAM clips=$n with_eou=$n_eou pack=$PACK"
cat <<EOC
# Italian gates (run from $MYNAH; the EOU pack ignores --lang in the CLI, lang_gate passes it anyway):
python3 tools/fetch_eval_bank.py --n 200 --langs it --out samples/eval-bank-it
python3 tools/eval/lang_gate.py -m $PACK --quant f32 --manifest samples/eval-bank-it/manifest.json \\
    --root samples/eval-bank-it --langs it --json .work/evidence/gate-eou-it.json
python3 tools/eval/eou_metrics.py -m $PACK --manifest samples/eval-bank-it/manifest.json \\
    --root samples/eval-bank-it --langs it --mode both --limit 60 --jobs 4 --json .work/evidence/eou-it.json
# baseline for the same bank: -m models/parakeet-realtime-eou-120m (English stock) -- latency/premature only
EOC
