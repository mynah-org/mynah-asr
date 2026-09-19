# Beyond Nemotron: what it takes to stream the other families

Status: OPEN (written 2026-09-19 on the dev host, from a full read of the
streaming path; no model was loaded — the models volume is not mounted here, so
everything below is a claim about the CODE, with file:line, and nothing here is
a measurement)

Task: M-1 (done), M-2, M-3, M-4, M-5
Question: the server holds several models in one fleet and serves Nemotron
streams well. Parakeet and Canary are in that fleet as offline groups. What
would it cost — in code and in cpu — to give them a WebSocket too, and which
part of "not Nemotron" is a defect versus a design boundary?

## The short answer

Three different things are being asked at once and they have three different
answers:

1. **Several models in one fleet** — done (S2-6). Parakeet and Canary already
   share the prefork router, the scheduler, the admission ladder and the
   batched step, as offline groups reached over REST. "A file is a slot fed
   faster than real time" is not a slogan here: it is the same slot, the same
   step, the same cancellation.
2. **Cache-aware streaming for a family that was trained cache-aware** — works
   today for any such pack, not only for the 3.5. The gate is a config
   property (`streaming.att_context_presets`), not a model name. The EN-only
   Nemotron with left context 70 needs no code, only a converter builder.
3. **Streaming a model that was NOT trained cache-aware** (every Parakeet,
   every Canary) — is not a gap in this runtime, it is a property of those
   checkpoints. It is possible (buffered/chunked streaming, §4) and it costs
   about **7x real time per stream**, against ~1.05x for Nemotron. That number,
   derived in §4, is the whole decision.

So the honest ordering is: Nemotron stays the streaming product; Parakeet and
Canary stay the offline half of the fleet, which is already served well;
buffered streaming is a deliberate, costed feature, not a missing branch.

## 1. The families, as the code sees them

| pack | attention | conv pad | conv norm | biases | mel norm | decoder | streams today |
|---|---|---|---|---|---|---|---|
| nemotron-3.5-streaming-0.6b | chunked_limited, left 56, right ∈ {0,1,3,6,13} | causal | layer_norm | none | NA | RNNT (LSTM 2x640) | **yes** |
| nemotron-speech-streaming-en-0.6b | chunked_limited, left **70** | causal | layer_norm | none | NA | RNNT | yes, once the converter has a builder for it |
| parakeet-tdt-0.6b-v3 / tdt-1.1b / rnnt-* | regular (full) | 'same' | batch_norm (folded) | none | per_feature | TDT / RNNT | no |
| parakeet-ctc-* | regular | 'same' | batch_norm | none | per_feature | CTC | no |
| parakeet-tdt_ctc-110m | regular | 'same' | batch_norm | **use_bias: true** | per_feature | TDT or CTC | no |
| canary-* | regular | 'same' | batch_norm | — | per_feature | AED (encoder-decoder) | no |

The offline path serves every row of that table. The **streaming** path
implements only the first two rows' worth of features — and until today it did
not say so.

## 2. What the streaming path assumes, line by line

Each of these is a branch the offline path has and the incremental path does
not. They are the reason a Parakeet pack with a hand-added `streaming` section
would produce *plausible wrong text* rather than an error.

| assumption | where | offline equivalent |
|---|---|---|
| no linear biases are added (q/k/v/o, both FFNs, both pointwise convs, the depthwise) | `src/encoder.c:932-957`, batched `1154-1211` | `src/encoder.c:418, 431, 453, 468-476` |
| no `xscale` on the layer input | absent from the step; offline at `src/encoder.c:1314-1315, 1372-1373` | ibid |
| conv norm is layer_norm, unconditionally | `src/encoder.c:891` | the BN-fold branch at `src/encoder.c:441-450` |
| the depthwise conv is causal (cache of `k-1` rows, taps `[0,k)`) | `src/encoder.c:873-890` | `pc = causal ? k-1 : (k-1)/2`, `src/encoder.c:428` |
| the mel is never normalised per feature | `src/features.c:229-272`, stated at `src/features.h:38-40` | `src/features.c:145-161` |
| the subsampling factor is 8 (three stride-2 stages) | `src/encoder.c:726` (`const int sub = 8`), `src/encoder.c:646`, `src/mynah_asr.c:761` | `encoder.subsampling_factor` in the pack |
| the subsampling stream stage pads (2,1) on the frequency axis whatever `ss->causal` says | `src/subsampling.c:370` | `pl = ss->causal ? 2 : 1`, `src/subsampling.c:238` |
| the decoder is RNNT/TDT, bypassing the engine vtable | `src/mynah_asr.c:860-862` calls `mynah_asr_greedy_decode_scratch` directly | `m->engine->decode`, `src/mynah_asr.c:485, 1350` |

**Closed on 2026-09-19**: `mynah_asr_stream_unsupported()`
(`src/mynah_asr.c`, public in `src/mynah_asr.h`) asks the LOADED WEIGHTS — a
bias tensor is present or it is not, `bn_fold` is allocated or it is not —
and refuses `mynah_asr_stream_open` with a named reason. The server calls it
before the WebSocket upgrade, so the answer is `400 model_not_streaming` with
the reason in the body (`server/sched.c`, `server/main.c`). No shipped pack can
trip it today: only Nemotron carries presets and Nemotron needs none of those
branches. That is exactly why it exists — it converts a future silent wrongness
into a refusal, and it is the precondition for everything below.

Two more, not on the streaming path but on the same theme (a model constant
that should be a config read):

- `src/decoder_ctc.c:34` takes the blank as `V - 1`, while the converter emits
  `decoder.blank_id` for both pure-CTC and hybrid packs
  (`tools/convert_nemo.py:347, 368`) and `mynah_asr_ctc_init` is never given
  it (`src/mynah_asr.c:243, 262`). True for today's packs, false in general.
- `src/vad.c:375` hardcodes `p.sample_rate = 16000` although the VAD pack
  carries the key (`tools/convert_silero.py:191`) and every other field is read
  from its JSON.

And in the server, the sample rate is a literal in six places
(`server/sched.c:730, 734, 816`; `server/main.c:420-422, 500, 661-667, 970`).
Harmless while every NeMo speech model is 16 kHz — and it is — but it is the
kind of thing that is cheap now and archaeological later.

## 3. What the stream state actually is

Per stream (`src/mynah_asr.c`, `src/encoder.c:655-703`):

- mel sliding window — generic, **correct only without per-feature
  normalisation**
- subsampling cache, `[3][C_in, F]`, one input frame per stage — causal
  geometry
- attention K/V cache, `[n_layers, left, d_model]` — `left` comes from the
  preset; with full attention (`left < 0`) the allocation is meaningless, which
  is why the preset gate is load-bearing
- conv cache, `[n_layers, k-1, d_model]` — exactly the causal left tail
- `cache_valid`, the rel-pos memo, the per-stream scratch
- decoder state (h/c/g/last_token/t_abs) — family-agnostic by construction
- detokeniser, VAD instance, ring buffers

Everything marked "causal" above is meaningful **only** for a windowed encoder.
That is the technical reason cache-aware streaming is not something one enables:
the caches are the model's training-time assumptions made resident.

## 4. Buffered streaming — the way a non-cache-aware model streams

The standard approach (what NeMo's `speech_to_text_streaming_infer_rnnt.py`
does): keep a ring of the last `L` seconds; every `P` seconds re-encode the
window `[left context | chunk | right context]`; decode only the frames that
belong to the chunk's centre, carrying the decoder state; drop the rest.

It needs **no server change at all**. The server's contract with a stream is
`need_samples()` → feed → step → deltas, and the cadence law
`T_step(B) = a + b·B ≤ ρ·P` does not care why the step is expensive. A buffered
model is a stream with a larger `P` (seconds, not 320 ms), a much larger `a`,
and its own group with its own cap. S2-6 already gives it that.

**The cost, which is the decision.** With NeMo's defaults (chunk 2 s, left
10 s, right 2 s) each step encodes 14 s of audio every 2 s:

```
compute per stream ≈ (left + chunk + right) / chunk = 14 / 2 = 7x real time
```

against ~1.05x for cache-aware Nemotron (the caches make each step cost one
chunk). So **one buffered Parakeet stream costs about what seven Nemotron
streams cost**, and the cap for its group has to be set accordingly. Shrinking
the left context cuts the cost linearly and the accuracy with it — that
trade-off has to be measured per model, never assumed.

Sizing that has to be checked on the box before any of this is worth building:
Parakeet 0.6B and Nemotron 0.6B have the same encoder shape, so the step cost
per encoded second should be within noise of each other, and the 7x is then a
straight multiplier on the measured `a` and `b`. If a Nemotron worker carries N
streams, the same worker carries about N/7 buffered Parakeet streams.

Three more things buffered streaming needs beyond the encoder:

- **A merge rule.** Text must be monotonic. For CTC/TDT the clean version is to
  decode only the centre frames and carry the decoder state, which never
  re-emits. LocalAgreement-style stabilisation is the alternative and it needs a
  protocol change (retraction), which v2 deliberately does not have —
  `is_final` is always true today (`src/mynah_asr.h:186-188`).
- **Per-feature mel normalisation over a window**, since these packs declare
  `normalize: per_feature` and the streaming mel does not normalise at all
  (`src/features.h:38-40`). Normalising over a sliding window is not the same
  arithmetic as normalising over the utterance, so this is a place where
  streaming ≡ offline byte-identity (repo rule 2) **cannot** hold, and the rule
  has to be restated as a tolerance for this mode rather than quietly broken.
- **Canary is a different problem entirely.** Its AED decoder allocates its
  caches inside the decode call and frees them at the end
  (`src/decoder_aed.c:154-161, 225`): there is no incremental state to carry. An
  AED model streams by re-decoding the window each time, which is a second
  expensive thing on top of the 7x encoder. Canary stays offline in v2, and
  that is a design boundary, not a TODO.

## 5. What this means for the board

Ordered by value per unit of risk:

1. **The guard** — done today, model-free, closes six silent-wrongness traps.
2. **Config reads that are currently literals** — CTC blank, VAD sample rate,
   subsampling factor in the three chunk-geometry sites, server sample rate.
   Small, testable against the offline path, and each one removes a lie.
3. **The converter builder for the EN-only Nemotron** — the only case where a
   *cache-aware* model is not served for a reason that is not architectural.
4. **Buffered streaming for Parakeet**, behind an explicit `stream_mode`, in
   its own group, with the 7x written into `docs/serving.md` before a line is
   written — and with the cap derived from a measured step, not from the
   cadence of a model it does not resemble.
5. Canary streaming: not in v2.

None of 2-5 is on the critical path for the box day. The box day is about
Nemotron on Linux, which is the product.

Evidence: (none — the models volume is not mounted on this host; every claim
above is a code reference)
Conclusion: (pending the cost check in §4 on the box)
Next action: board item for §5.2, after the box day.
