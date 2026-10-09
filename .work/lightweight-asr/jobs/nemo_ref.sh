#!/bin/bash
# NeMo reference for the EOU 120M on the SAME FLEURS clips as q1 (S15-2 gate 6:
# reference parity). Separate venv so tools/ stays untouched.
#   tmux new -d -s nemoref 'bash /root/nemo_ref.sh 2>&1 | tee /root/res/nemo_ref.log'
set -u
V=/root/nemo-venv; O=/root/res/nemo_ref; mkdir -p $O
if [ ! -x $V/bin/python ]; then
    uv venv -q -p 3.12 $V && VIRTUAL_ENV=$V timeout 2400 uv pip install -q "nemo_toolkit[asr]>=2.6" 2>&1 | tail -3
fi
$V/bin/python -c "import nemo; print('nemo', nemo.__version__)" || exit 3
cd /root/mynah-asr || exit 2
CUDA_VISIBLE_DEVICES= timeout 3600 $V/bin/python /root/nemo_ref.py /root/res/q1/eou-en.json samples/eval-bank $O 2>&1 | grep -v -E "^\[NeMo W|warnings.warn|^\s*$" | tail -40
echo "== NEMOREF-DONE $(date +%T)"
