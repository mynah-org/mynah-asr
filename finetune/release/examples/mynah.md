# Running the model with Mynah (mynah-asr, C / CUDA)

Build Mynah (`make`, and `make -C gpu CUDA_ARCH=sm_89` for the CUDA server), then point it at the `mynah/` directory
of this package (model.safetensors, mynah.json, tokens.json, mel_filters.safetensors).

CPU, one file, streaming with published deltas (16 kHz mono PCM16 WAV):

    MYNAH_ASR_THREADS=4 ./mynah-asr stream -m mynah/ -i audio.wav --quant f32 --deltas

Each JSON line is a delta; `<EOU>` events end a segment. Raw s16le 16 kHz mono on stdin also works (omit `-i`).

CUDA WebSocket server (bf16 by default; `--precision f32` for bit-parity with the CPU engine):

    ./mynah-asr-server-cuda -m mynah/ -p 8291        # health: curl localhost:8291/v1/health
    # stream: WebSocket ws://localhost:8291/v1/audio/stream, binary frames of PCM16 16 kHz mono;
    # server messages: {"type": "delta", ...}, {"type": "eou", "source": "model", ...}, {"type": "final", ...}

Semantics: after a model `<EOU>` the runtime resets the encoder caches and the decoder after the chunk holding it.
Size the CPU thread count to the CPU quota actually available (containers: the cgroup cpu.max, not nproc).
