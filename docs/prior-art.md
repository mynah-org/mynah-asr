# Mynah — Prior art (studied on the real code, 2026-07-16)

Two projects analyzed in depth by cloning the sources. Strategic note: **parakeet.cpp already
supports nemotron-3.5-asr-streaming-0.6b** (offline + cache-aware streaming, WER 0 vs NeMo on
5 languages). We are not starting from scratch conceptually: we have a complete map of the pitfalls.
Mynah's differentiators remain: pure C without ggml (own kernels, qwen-* style), latency profile
selectable at runtime (parakeet.cpp writes it into the GGUF but doesn't read it!), integrated
OpenAI-compatible server, multi-engine design from the start.

---

## A. mudler/parakeet.cpp (C++17 on ggml, GGUF)

### To IMITATE

1. **Verbatim NeMo tensor names + everything metadata-driven**: the converter renames nothing;
   ~40 KV keys (`arch, dims, mel params, vocab, max_symbols, att_context_*, streaming.*,
   prompt.*`) drive the runtime. This is the decision that let it cover 12 checkpoints
   almost without touching C++. → Mynah: same principle with our `config.json` + safetensors.
2. **Filterbank and Hann window lifted from the checkpoint** (`preprocessor.featurizer.fb` /
   `.window` exported as-is) → exact mel parity for free, no slaney formula to replicate
   by hand. Our converter must do the same (or generate them with the onnx-asr
   formula and validate them, atol 5e-7).
3. **3-level validation methodology** (our test plan is ready):
   - `gen_nemo_baseline.py`: dump of intermediate NeMo tensors (mel, subsampling, layer
     0/mid/last, encoder_out, pred_out, joint_out) with dither=0/no_grad/eval → per-stage
     C tests with tolerances 4e-5…6e-3;
   - `validate_vs_nemo.py`: end-to-end WER (exit 0/1, **77 = skip if the checkpoint is missing**
     → CI always green);
   - dedicated streaming baselines: token-by-token vs NeMo's `cache_aware_stream_step`,
     including the reset-on-EOU case with a two-utterance clip.
4. **Exact, already-verified streaming semantics** (to port as-is):
   - `cache_last_time[layer]`: `[K−1=8, d_model]`, left context of the depthwise conv;
   - `cache_last_channel[layer]`: `[cache_len=56, d_model]`, fed with **the attention's
     post-norm input** (not the output); `clc_len` = valid frames, empty columns
     masked;
   - `drop_extra_pre_encoded`: after subsampling the chunk (which includes the pre-encode
     overlap) the redundant leading frames are discarded (only from chunk ≥1);
   - mid-stream keep `valid_out_len` frames, the last chunk keeps everything;
   - pre-encode overlap: built by the caller (chunk n = overlap + new frames);
   - the chunk that touches the end of the buffer is **deferred to finalize** (`is_last=true`).
5. **Optimized greedy RNNT** (`rnnt.cpp`, to port almost line by line):
   - `enc_proj` precomputed over all frames in one matmul;
   - **pred-net output cache**: the LSTM is recomputed only on a non-blank emission
     (~U times instead of T+U);
   - argmax on raw logits (no softmax);
   - `RnntDecodeState` lifted and **chunk-invariant**: incremental decoding ≡ whole →
     streaming decode becomes trivial.
6. **Numerical details that buy parity**: double in mel/FFT/pos-enc; constant pad
   (NOT reflect); std ddof=1 + eps 1e-5 (when normalize is present); batch_norm folded host-side
   into scale/shift; PyTorch LSTM gate order `[i,f,g,o]` with both biases; SOS = zero embedding
   of the blank; `T = 1+floor(S/hop)`; valid offline = T−1 (center-pad adds one frame).
7. **Frame-local StreamingMel** (possible ONLY with normalize=NA — our case):
   preemphasis carried across feeds, O(n_fft) buffer, tail emitted at finalize.
8. **C-API**: opaque handle, `abi_version()`, per-ctx `last_error`, typed EOU/EOB events
   + bitmask, `finalize` that does not fabricate an EOU. Design to copy in `mynah_asr.h`.
9. **Quantization**: only the large linears that enter the matmul (FFN, attn q/k/v/out/pos,
   pre_encode.out, joint enc/pred = 90%+ of the weights); conv/LSTM/bias/norm in F32. Evidence:
   WER 0 down to q4_k. → our INT8 (M5) can be just as surgical.
10. **Known pitfall (issue #13)**: after `<EOU>` only the **decoder** must be reset (LSTM to
    zero, last_token→SOS), NOT the encoder cache — otherwise the stream goes mute. Relevant
    for the EOU models (parakeet_realtime_eou); base Nemotron 3.5 has no EOU but the
    reset-decoder-only pattern should be kept.

### To do DIFFERENTLY (in pure C)

1. **No per-call graph building**: parakeet rebuilds thousands of ggml nodes and does
   string lookups of the weights on every chunk. → Mynah: weight pointers resolved ONCE
   at load into per-layer structs, direct kernels.
2. **LSTM and joint step as direct kernels** (matvec 4H×H + gates); the bottleneck
   is the joint head's 640→13088 matmul, to parallelize over rows.
3. **Depthwise conv and subsampling without im2col**: in C a causal k=9 loop is trivial and
   cache-friendly; the `pad_ext` workaround for asymmetric padding also disappears.
4. **Attention without a materialized mask**: the chunked_limited window is expressed in the
   loop bounds (start/end of kj for each qi); `rel_shift` is done by indexing
   `bd[qi,kj] = qv[qi]·p[pos_of(qi,kj)]` — zero pad/reshape/view acrobatics, zero O(T²)
   memory per layer.
5. **A single path**: no dual scalar/batched code path (half of their code is
   B=1 vs B>1 duplication) — v1 streaming is B=1; batching arrives with the server (M4).
6. **Expose the right-context choice at runtime** (`--latency 80|160|320|560|1120`):
   they ignore it, for us it is an easy differentiator.
7. Time-major mel layout or ring buffer (their `append_mel_frames` rebuilds the buffer
   on every feed, cumulative O(T²)).

---

## B. istupakov/onnx-asr (Python numpy + onnxruntime, no torch)

Best foundation for **our oracle** (`tools/oracle/`): readable, already validated vs NeMo.

### To reuse almost verbatim
1. **Exact NeMo preprocessing in numpy** (`preprocessors/numpy_preprocessor.py:139-182`):
   preemph 0.97 (first sample unchanged) → constant pad n_fft/2 per side → framing 512/160
   → **symmetric** Hann 400 zero-padded to 512 → rfft → |X|² → `@ fbanks` → `log(x+2^-24)`
   → (for us: STOP, normalize NA). `features_lens = len // hop`.
2. **Slaney mel filterbank** (`preprocessors/fbanks.py:27-57`): clone of torchaudio
   (linear <1000 Hz, log above, slaney norm `2/(m[2:]-m[:-2])`), computed in float64.
   Validated vs NeMo with atol 5e-7.
3. **Greedy transducer state machine** (`asr.py:192-229`): SOS=blank, LSTM state
   **committed only on non-blank tokens**, t advances on blank or on max_tokens_per_step,
   TDT: `t += step` with guard `step==0 && blank → t+=1`.
4. **Test tolerances**: features vs NeMo atol 5e-4 rtol 1e-4; filterbank 5e-7 —
   reference for our C↔oracle comparisons.
5. Pure numpy `read_wav` (PCM 8/16/24/32) and timestamps `0.01 × subsampling × frame_idx`.

### Limitations (what NOT to take)
- No streaming/cache-aware (the main gap); no parametrized normalize=NA.
- Naive tokenizer (space-separated vocab.txt, no byte fallback) — we decode
  from the real `tokenizer.model`/`tokenizer.json`.
- TDT: uses the argmax index as the duration (fine only for identity durations [0..4]).
- Monolithic ONNX decoder_joint → recomputes the pred-net even on blanks (us: separate + cache).

---

## C. Decisions for Mynah derived from the prior art

| Decision | Source |
|---|---|
| Converter: verbatim HF tensor names, everything in the Mynah `config.json` (KV-style) | parakeet.cpp |
| Filterbank+window exported from the checkpoint/processor into the converted model | parakeet.cpp |
| Oracle: fork of onnx-asr's numpy preprocessing (without normalize) + HF forward | onnx-asr |
| Per-stage baselines in files (mel, subsampling, layer 0/12/23, enc_out, pred, joint) | parakeet.cpp |
| Model-dependent tests: skip with exit 77 if the checkpoint is missing | parakeet.cpp |
| Greedy RNNT: precomputed enc_proj + pred-net cache + raw logits + chunk-invariant state | both |
| Streaming: exact NeMo cache semantics (see §A.4 above) | parakeet.cpp |
| Frame-local streaming mel (possible with normalize=NA) | parakeet.cpp |
| Quantization (M5): only large linears, rest in F32 | parakeet.cpp |
| Runtime-selectable latency (differentiator) | parakeet.cpp gap |
