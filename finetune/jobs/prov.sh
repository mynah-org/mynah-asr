#!/bin/bash
# Provision a fresh Linux CUDA box for the lightweight-ASR track and its fine-tuning (finetune/).
# Restartable: every step leaves a marker in $R/done/ and is skipped next time.
# Everything is downloaded ON THE BOX (HF + FLEURS); nothing comes from the dev machine.
# The HF token is read from /root/.hf_token (mode 600); KEEP_TOKEN=1 keeps it for the
# day's archival uploads (the close-out job deletes it), otherwise it is deleted at the end.
# Also builds the NeMo fine-tuning venv (/root/nemo-venv: nemo_toolkit[asr], torch cu128
# via canary/setup.sh, numba-cuda + numpy<2.4 for the warprnnt_numba RNNT loss, torchaudio
# of the SAME torch build: nemo_toolkit[asr] does not pull it, prepare_it's resampler needs it) in the
# background of the build. A fresh 32 GB L40S is ready in ~20-30 min.
#   tmux new -d -s prov 'KEEP_TOKEN=1 bash /root/prov.sh 2>&1 | tee -a /root/res/prov.log'
set -u
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
mkdir -p /root/ft/models /root/ft/logs /root/ft/runs /root/eou-kit
R=/root/res; mkdir -p $R/done
REPO=/root/mynah-asr; BRANCH=${BRANCH:-lw-finetune-tooling}
step() { [ -e "$R/done/$1" ] && { echo "== skip $1"; return 1; }; echo "== $1 $(date +%T)"; return 0; }
ok() { touch "$R/done/$1"; echo "== ok $1 $(date +%T)"; }
hf() {  # hf <repo> <file> <dest>: resumable, authenticated
    curl -fL --retry 5 -C - -H "Authorization: Bearer $(cat /root/.hf_token)" \
        -o "$3" "https://huggingface.co/$1/resolve/main/$2"
}

if step clone; then
    [ -d $REPO ] || git clone -q https://github.com/mynah-org/mynah-asr $REPO
    git -C $REPO fetch -q origin && git -C $REPO checkout -q $BRANCH && git -C $REPO pull -q --ff-only && ok clone
fi
ln -sfn $REPO /root/mynah-eou   # the job scripts address the EOU checkout by this name
cd $REPO || exit 2
V=/root/nemo-venv
if step venv; then
    ( uv venv -q -p 3.12 $V && VIRTUAL_ENV=$V timeout 2400 uv pip install -q "nemo_toolkit[asr]>=2.6" \
      && FT_ROOT=/root/ft VENV=$V timeout 3000 bash finetune/canary/setup.sh \
      && VIRTUAL_ENV=$V uv pip install -q numba-cuda "numpy<2.4" silero-vad \
      && VIRTUAL_ENV=$V uv pip install -q --index-url https://download.pytorch.org/whl/cu128 \
           --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
           "torchaudio==$($V/bin/python -c 'import torch; print(torch.__version__)')" \
      && $V/bin/python -c "import torch, torchaudio, scipy.signal; assert torch.cuda.is_available()" \
      && FT_ROOT=/root/ft $V/bin/python finetune/eou/prepare_mix.py --selftest \
      && cp finetune/eou/two_stage.py /root/eou-kit/plain_it2.py \
      && cp finetune/eou/export_to_mynah.sh /root/eou-kit/ && ok venv ) > $R/venv.log 2>&1 &
    VPID=$!
fi
if step build; then
    timeout 1800 make -j32 lib >$R/build.log 2>&1 && timeout 1800 make -j32 all >>$R/build.log 2>&1 \
      && timeout 2400 make -C gpu -j32 CUDA_ARCH=sm_89 >>$R/build.log 2>&1 && ok build || tail -20 $R/build.log
fi
if step pyenv; then
    (cd tools && timeout 1800 uv sync --extra oracle -q) && ok pyenv
fi
# .nemo models (converted in place by tools/convert_nemo.py)
for m in "parakeet-realtime-eou-120m|nvidia/parakeet_realtime_eou_120m-v1|parakeet_realtime_eou_120m-v1.nemo" \
         "canary-180m-flash|nvidia/canary-180m-flash|canary-180m-flash.nemo"; do
    d=${m%%|*}; rest=${m#*|}; repo=${rest%%|*}; f=${rest#*|}
    if step "model-$d"; then
        mkdir -p models/$d
        hf $repo $f models/$d/$f && (cd tools && timeout 1800 uv run python convert_nemo.py ../models/$d >$R/convert-$d.log 2>&1) \
          && ok "model-$d" || tail -5 $R/convert-$d.log
    fi
done
# Nemotron: HF-native port
if step model-nemotron; then
    timeout 3600 scripts/download_model.sh --model nemotron >$R/dl-nemotron.log 2>&1 \
      && (cd tools && timeout 1800 uv run python convert_nemo.py ../models/nemotron-3.5-asr-streaming-0.6b >$R/convert-nemotron.log 2>&1) \
      && ok model-nemotron || tail -5 $R/dl-nemotron.log $R/convert-nemotron.log
fi
[ -n "${KEEP_TOKEN:-}" ] || { rm -f /root/.hf_token; echo "== token removed"; }
# Silero VAD: the ONNX AND its Mynah conversion (eou_metrics.py SKIPs, rc 77, without it)
if step vad; then make fetch-vad >$R/vad.log 2>&1 \
    && (cd tools && timeout 900 uv run --extra vad python convert_silero.py ../models/silero-vad/silero_vad.onnx ../models/silero-vad >>$R/vad.log 2>&1) \
    && ok vad || tail -5 $R/vad.log; fi
# the CUDA engine parity/serving gate binary used by jobs/m1.sh (not part of `make -C gpu`)
if step gpu-tests; then timeout 1800 make -C gpu -j32 CUDA_ARCH=sm_89 test-stream >$R/gpu-tests.log 2>&1 && ok gpu-tests || tail -5 $R/gpu-tests.log; fi
# Banks (FLEURS, CC-BY 4.0, public)
if step bank-stress; then
    (cd tools && timeout 3600 uv run python fetch_stress_bank.py >$R/bank-stress.log 2>&1) && ok bank-stress || tail -5 $R/bank-stress.log
fi
if step bank-eval; then
    (cd tools && timeout 3600 uv run python fetch_eval_bank.py --n 200 >$R/bank-eval.log 2>&1 \
      && timeout 3600 uv run python fetch_eval_bank.py --n 200 --langs it --out ../samples/eval-bank-it >>$R/bank-eval.log 2>&1) \
      && ok bank-eval || tail -5 $R/bank-eval.log
fi
[ -n "${VPID:-}" ] && wait $VPID
[ -e $R/done/venv ] || { echo "== venv FAILED"; tail -20 $R/venv.log; }
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; df -h /root | tail -1
ls $R/done; echo "== PROV-DONE $(date +%T)"
