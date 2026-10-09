#!/bin/bash
# Step 1: make /root/nemo-venv (NeMo 3.0.0, created with `uv venv -p 3.12` +
# `uv pip install nemo_toolkit[asr]`) usable on a driver-575 box (CUDA 12.9
# max): the default PyPI torch wheel targets a newer CUDA runtime than the
# driver supports, so torch is replaced by the SAME version built for cu128
# (fallback cu126; NeMo 3.0.0 requires torch>=2.6.0 and its own cu12 extra
# pins torch==2.12.0+cu126). Then: extra deps, the leaderboard normaliser
# (pinned commit), the Canary 180M checkpoint, and a GPU/bf16 check.
#   tmux new -d -s ft 'bash setup.sh 2>&1 | tee -a /root/ft/logs/setup.log'
. "$(dirname "$0")/common.sh"
[ -x "$PY" ] || { echo "no venv python at $PY (set VENV=...)"; exit 2; }

cuda_ok() { "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

if step setup-torch; then
    cur=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo none)
    echo "   installed torch: $cur"
    if cuda_ok && [ -z "${FORCE_TORCH:-}" ]; then
        echo "   torch already sees the GPU; keeping it"
    else
        base=${cur%%+*}
        for cu in ${TORCH_CUDA:-cu128 cu126}; do
            vers=$(curl -fsSL "https://download.pytorch.org/whl/$cu/torch/" \
                | grep -o "torch-[0-9][0-9.]*%2B$cu-cp312-cp312-manylinux[^\"#]*x86_64\.whl" \
                | sed 's/^torch-\([0-9.]*\)%2B.*/\1/' | sort -uV)
            [ -n "$vers" ] || { echo "   no cp312 wheels listed for $cu"; continue; }
            if [ -n "${TORCH_VERSION:-}" ]; then want=$TORCH_VERSION
            elif echo "$vers" | grep -qx "$base"; then want=$base
            else want=$(echo "$vers" | tail -1); fi
            echo "   installing torch==$want+$cu (available on $cu: $(echo "$vers" | tr '\n' ' '))"
            pk=("torch==$want+$cu")
            if "$PY" -c "import torchaudio" 2>/dev/null; then
                if curl -fsSL "https://download.pytorch.org/whl/$cu/torchaudio/" | grep -q "torchaudio-$want%2B$cu-cp312"; then
                    pk+=("torchaudio==$want+$cu")
                else echo "   WARNING: no torchaudio $want for $cu; leaving torchaudio as is"; fi
            fi
            timeout 2400 uv pip install --python "$PY" --index-url "https://download.pytorch.org/whl/$cu" \
                --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
                --reinstall-package torch "${pk[@]}" && cuda_ok && break
            echo "   $cu did not give a working CUDA torch; trying the next"
        done
        cuda_ok || fail setup-torch 3
    fi
    ok setup-torch
fi

if step setup-deps; then
    timeout 1200 uv pip install --python "$PY" num2words regex soundfile pyarrow sentencepiece protobuf huggingface_hub \
        || fail setup-deps $?
    ok setup-deps
fi

if step setup-normalizer; then
    # HF Open ASR Leaderboard normaliser, pinned (Apache-2.0; Whisper-derived)
    sha=${OAL_SHA:-67e8bd6acea240819ad67080f6f31e15d4a90da5}
    d=$FT/vendor/oal_norm; mkdir -p "$d"; : >"$d/__init__.py"
    for f in normalizer.py english_abbreviations.py; do
        curl -fsSL --retry 5 -o "$d/$f" "https://raw.githubusercontent.com/huggingface/open_asr_leaderboard/$sha/normalizer/$f" \
            || fail setup-normalizer $?
    done
    "$PY" -c "import sys; sys.path.insert(0,'$KIT'); import ftlib; r=ftlib.selfcheck_normalizer(); print('   normaliser self-check:', r); sys.exit(0 if r=='ok' else 1)" \
        || fail setup-normalizer 5
    ok setup-normalizer
fi

if step setup-model; then
    mkdir -p "$FT/models"; m=$FT/models/canary-180m-flash.nemo
    local_copy=/root/mynah-asr/models/canary-180m-flash/canary-180m-flash.nemo
    if [ -s "$local_copy" ]; then ln -sf "$local_copy" "$m"
    else timeout 1800 curl -fL --retry 5 -C - -o "$m" https://huggingface.co/nvidia/canary-180m-flash/resolve/main/canary-180m-flash.nemo \
        || fail setup-model $?; fi
    ls -lL "$m"
    ok setup-model
fi

echo "== verify $(date +%T)"
timeout 900 "$PY" - <<'PY' || fail verify $?
import time, torch
print("   torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available"
print("   gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0),
      "bf16", torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), "bf16 not supported"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): y = x @ x
torch.cuda.synchronize(); dt = time.time() - t
print(f"   bf16 matmul ok: {20 * 2 * 4096**3 / dt / 1e12:.1f} TFLOP/s")
with torch.autocast("cuda", dtype=torch.bfloat16):
    z = torch.nn.functional.linear(torch.randn(8, 512, device="cuda"), torch.randn(256, 512, device="cuda"))
assert z.dtype == torch.bfloat16
import nemo, lightning
from nemo.collections.asr.models import EncDecMultiTaskModel  # noqa: F401
print("   nemo", nemo.__version__, "lightning", lightning.__version__)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "   disk free on $FT: $(free_gb) GB"
echo "== SETUP-DONE $(date +%T)"
