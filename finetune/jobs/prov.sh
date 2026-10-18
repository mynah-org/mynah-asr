#!/bin/bash
# Provision a fresh Linux CUDA box for the lightweight-ASR track and its fine-tuning (finetune/).
# Restartable: every step leaves a marker in $R/done/ and is skipped next time.
# Everything is downloaded ON THE BOX (HF + FLEURS); nothing comes from the dev machine.
# The HF token is read from /root/.hf_token (mode 600); KEEP_TOKEN=1 keeps it for the
# day's archival uploads (the close-out job deletes it), otherwise it is deleted at the end.
# Also builds the NeMo fine-tuning venv (/root/nemo-venv: nemo_toolkit[asr], torch cu128
# via canary/setup.sh, numba-cuda + numpy<2.4 for the warprnnt_numba RNNT loss, torchaudio
# of the SAME torch build: nemo_toolkit[asr] does not pull it, prepare_it's resampler needs it) in the
# background of the build. torch is pinned to 2.11.0+cu128, the build of every 2026-10 run: on a CUDA 13
# driver setup.sh would keep PyPI's torch 2.14.1+cu130, which has no cu128 torchaudio (and changes the stack).
# numba-cuda's RNNT kernels are linked at run time: NVVM and nvJitLink must be ONE CUDA release. Since 2026-10-15
# numba-cuda pulls an NVVM 12.9 while torch cu128 pins nvJitLink 12.8 -> "nvvmAddNVVMContainerToProgram ... may need
# newer version of nvJitLink" at the first RNNT loss; the 12.9 toolchain is pinned (also needed for sm_121 / GB10).
# A fresh 32 GB L40S is ready in ~20-30 min.
#   tmux new -d -s prov 'KEEP_TOKEN=1 bash /root/prov.sh 2>&1 | tee -a /root/res/prov.log'
set -u
# TLS gate BEFORE the token is ever sent: on 2026-10-13 a host intercepted HTTPS (HF, PyPI, PyTorch) with its own
# "Internal Proxy Root CA" injected into the container trust store, so the Bearer token was readable on the host.
# Every endpoint must chain to a public CA; otherwise the token is deleted and provisioning stops.
for h in huggingface.co cdn-lfs.hf.co files.pythonhosted.org download.pytorch.org; do
    iss=$(echo | timeout 15 openssl s_client -connect $h:443 -servername $h 2>/dev/null | openssl x509 -noout -issuer 2>/dev/null)
    echo "== tls $h: ${iss:-no certificate}"
    echo "$iss" | grep -qE "O ?= ?(Amazon|Google Trust Services|GlobalSign|DigiCert|Let's Encrypt|Sectigo|Fastly|Cloudflare)" || {
        rm -f /root/.hf_token; echo "== TLS GATE FAILED on $h (non-public issuer): token deleted, stop"; exit 9; }
done
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; }
export PATH=$HOME/.local/bin:$PATH
# A small /dev/shm (some hosts cap it at 31-41 GB and refuse a remount) cannot hold A2 + the Granary pools +
# clean-it: then every /dev/shm audio dir the jobs use is a symlink to the root disk, so manifest paths (and the
# archived teacher scores keyed on them) stay identical. SHM_MIN_GB (default 120) is the threshold.
if [ "$(df -Pk /dev/shm | awk 'NR==2 {print int($2 / 1048576)}')" -lt "${SHM_MIN_GB:-120}" ]; then
    for d in clean-it ft-audio mix yodas; do
        mkdir -p /root/disk/$d
        [ -L /dev/shm/$d ] || { rmdir /dev/shm/$d 2>/dev/null; [ -e /dev/shm/$d ] || ln -s /root/disk/$d /dev/shm/$d; }
    done
    echo "== small /dev/shm: audio dirs on the root disk ($(ls -l /dev/shm | grep -c -- '->') symlinks)"
fi
mkdir -p /root/ft/models /root/ft/logs /root/ft/runs /root/eou-kit /dev/shm/ft-audio
# MLS audio (canary/prepare_it.sh writes $FT/audio) lives in RAM: the 32 GB root disk holds the venvs,
# models and checkpoints and cannot also take the 40 h + eval audio (need_gb 12 failed on 2026-10-11)
[ -e /root/ft/audio ] || ln -s /dev/shm/ft-audio /root/ft/audio
R=/root/res; mkdir -p $R/done
REPO=/root/mynah-asr; BRANCH=${BRANCH:-lw-finetune-tooling}
step() { [ -e "$R/done/$1" ] && { echo "== skip $1"; return 1; }; echo "== $1 $(date +%T)"; return 0; }
ok() { touch "$R/done/$1"; echo "== ok $1 $(date +%T)"; }
hf() {  # hf <repo> <file> <dest>: resumable, authenticated (token via a curl config on a pipe, never in argv)
    curl -fL --retry 5 -C - -K <(printf 'header = "Authorization: Bearer %s"\n' "$(cat /root/.hf_token)") \
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
      && FT_ROOT=/root/ft VENV=$V FORCE_TORCH=${FORCE_TORCH-1} TORCH_VERSION=${TORCH_VERSION-2.11.0} TORCH_CUDA=${TORCH_CUDA-cu128} \
         timeout 3000 bash finetune/canary/setup.sh \
      && VIRTUAL_ENV=$V uv pip install -q numba-cuda "numpy<2.4" silero-vad \
      && VIRTUAL_ENV=$V uv pip install -q "nvidia-cuda-nvcc-cu12==12.9.86" "nvidia-cuda-nvrtc-cu12==12.9.86" \
           "nvidia-cuda-runtime-cu12==12.9.79" "nvidia-nvjitlink-cu12==12.9.86" \
      && VIRTUAL_ENV=$V uv pip install -q --index-url https://download.pytorch.org/whl/cu128 \
           --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
           "torchaudio==$($V/bin/python -c 'import torch; print(torch.__version__)')" \
      && $V/bin/python -c "import torch, torchaudio, scipy.signal; assert torch.cuda.is_available()" \
      && $V/bin/python -c "
import torch
from nemo.collections.asr.losses.rnnt import RNNTLoss
lp = torch.randn(2, 50, 11, 1027, device='cuda', requires_grad=True)
out = RNNTLoss(num_classes=1026, loss_name='warprnnt_numba')(log_probs=lp, targets=torch.randint(0, 1026, (2, 10), device='cuda'),
    input_lengths=torch.full((2,), 50, device='cuda'), target_lengths=torch.full((2,), 10, device='cuda'))
out.sum().backward(); print('rnnt loss gate ok', float(out))" \
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
# Canary 180M (1.4 GB) only with CANARY=1: the EOU fine-tuning days do not use it
for m in "parakeet-realtime-eou-120m|nvidia/parakeet_realtime_eou_120m-v1|parakeet_realtime_eou_120m-v1.nemo" \
         ${CANARY:+"canary-180m-flash|nvidia/canary-180m-flash|canary-180m-flash.nemo"}; do
    d=${m%%|*}; rest=${m#*|}; repo=${rest%%|*}; f=${rest#*|}
    if step "model-$d"; then
        mkdir -p models/$d
        hf $repo $f models/$d/$f && (cd tools && timeout 1800 uv run python convert_nemo.py ../models/$d >$R/convert-$d.log 2>&1) \
          && ok "model-$d" || tail -5 $R/convert-$d.log
    fi
done
# two_stage.py's default --stock and the jobs' STOCK path
[ -e models/parakeet-realtime-eou-120m/parakeet_realtime_eou_120m-v1.nemo ] && \
    ln -sfn $REPO/models/parakeet-realtime-eou-120m/parakeet_realtime_eou_120m-v1.nemo /root/ft/models/parakeet_realtime_eou_120m-v1.nemo
# Nemotron: HF-native port
if step model-nemotron; then
    timeout 3600 scripts/download_model.sh --model nemotron >$R/dl-nemotron.log 2>&1 \
      && (cd tools && timeout 1800 uv run python convert_nemo.py ../models/nemotron-3.5-asr-streaming-0.6b >$R/convert-nemotron.log 2>&1) \
      && ok model-nemotron || tail -5 $R/dl-nemotron.log $R/convert-nemotron.log
fi
# the NeMo .nemo of the same teacher: encoder KD (e1_kd.sh / kd_audit.py TEACHER) loads it with NeMo
if step teacher-nemo; then
    hf nvidia/nemotron-3.5-asr-streaming-0.6b nemotron-3.5-asr-streaming-0.6b.nemo /root/ft/models/nemotron-3.5-asr-streaming-0.6b.nemo \
      && ok teacher-nemo
fi
[ -n "${KEEP_TOKEN:-}" ] || { rm -f /root/.hf_token; echo "== token removed"; }
# Silero VAD: the ONNX AND its Mynah conversion (eou_metrics.py SKIPs, rc 77, without it)
if step vad; then make fetch-vad >$R/vad.log 2>&1 \
    && (cd tools && timeout 900 uv run --extra vad python convert_silero.py ../models/silero-vad/silero_vad.onnx ../models/silero-vad >>$R/vad.log 2>&1) \
    && ok vad || tail -5 $R/vad.log; fi
# the CUDA engine parity/serving gate binary used by jobs/m1.sh (not part of `make -C gpu`)
if step gpu-tests; then timeout 1800 make -C gpu -j32 CUDA_ARCH=sm_89 test-stream >$R/gpu-tests.log 2>&1 && ok gpu-tests || tail -5 $R/gpu-tests.log; fi
# Banks (FLEURS, CC-BY 4.0, public)
if [ -n "${STRESS:-}" ] && step bank-stress; then   # 641 MB, serving benches only (STRESS=1)
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
