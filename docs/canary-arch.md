# Canary Flash — verified architecture (AED)

Source: `model_config.yaml` + `model_weights.ckpt` from the official `.nemo` files (2026-07-18)
and NeMo sources (`canary2.py`, `transformer_modules.py`, `transformer_decoders.py`,
`aed_multitask_models.py`). M6 targets: `nvidia/canary-180m-flash` (de-risking) and
`nvidia/canary-1b-flash`. NeMo class: `EncDecMultiTaskModel` (multitask AED:
ASR + speech translation en↔de/es/fr).

NOTE: the HF-native export of canary-1b-flash (config.json + model.safetensors) is
ENCODER-ONLY (1292 `encoder.*` tensors, `nemo_decoder_type: none`): for Canary we
ALWAYS go through the `.nemo` (extractor already in convert_nemo.py, extended for the AED).

## 180m-flash by the numbers

| block | value |
|---|---|
| mel | 128 bins, per_feature, n_fft 512, win 400, hop 160, dither 0 (inference) |
| encoder | FastConformer **17L·512·8h**, ff 4×, k9 **batch_norm**, dw_striding 8× **non-causal**, att **full [-1,-1]**, rel_pos, xscaling false, use_bias true (default) |
| projection | `encoder_decoder_proj`: Linear 512→1024 (+bias). Identity if enc==dec hidden |
| transf_encoder | 0 layers (absent from the ckpt: not used) |
| transf_decoder | Transformer **4L·1024·8h**, inner 4096, **ReLU**, **pre-LN** + final_layer_norm, max_seq 1024 |
| head | `log_softmax.mlp.layer0`: Linear 1024→5248 (softmax not needed for greedy) |
| vocab | 5248 = spl_tokens 1152 + en/de/es/fr 1024 each (aggregate, cumulative offsets) |

1b-flash: encoder 32L·1024 (proj = Identity, hence absent from the ckpt), decoder
4L·1024, vocab 16384? (verify from the yaml at conversion time). Same structure.

The encoder is EXACTLY the non-causal Parakeet variant already implemented
(folded BN, full attention, bias, symmetric subsampling) — full reuse,
Metal included. The new work is only proj + decoder + tokenizer + prompt.

## Transformer decoder (ckpt names → canonical)

```
encoder_decoder_proj.{weight,bias}                    → enc_dec_proj.*   [1024,512]
transf_decoder._embedding.token_embedding.weight      → aed.embedding.weight [5248,1024]
transf_decoder._embedding.position_embedding.pos_enc  → aed.pos_enc [1024,1024] (buffer!)
transf_decoder._embedding.layer_norm.{weight,bias}    → aed.emb_norm.*
transf_decoder._decoder.layers.N.layer_norm_1.*       → aed.layers.N.ln_self.*
  .first_sub_layer.{query,key,value}_net, out_projection → aed.layers.N.self_attn.{q,k,v,o}_proj
transf_decoder._decoder.layers.N.layer_norm_2.*       → aed.layers.N.ln_cross.*
  .second_sub_layer....                               → aed.layers.N.cross_attn.{q,k,v,o}_proj
transf_decoder._decoder.layers.N.layer_norm_3.*       → aed.layers.N.ln_ffn.*
  .third_sub_layer.dense_in/dense_out                 → aed.layers.N.ffn.linear1/linear2
transf_decoder._decoder.final_layer_norm.*            → aed.final_norm.*
log_softmax.mlp.layer0.{weight,bias}                  → aed.head.*
```

Semantics verified against the NeMo sources:

- **Embedding**: `emb = LayerNorm(token_emb[id] + pos_enc[pos])` then dropout (0 in
  inference). NO sqrt(d) scaling on the token embedding; the scale is already INSIDE
  `pos_enc` (buffer in the ckpt: sin/cos **divided by sqrt(hidden)** — take it
  from the ckpt, bit-exact, like the mel filterbanks). Positions start at 0 (start_pos for
  incremental generation).
- **Pre-LN block** (`forward_preln`): `x += SelfAttn(LN1(x))` (causal mask) →
  `x += CrossAttn(LN2(x), enc)` → `x += FFN(LN3(x))`; after the last layer,
  `final_layer_norm`.
- **Attention** (self and cross): q and k are EACH divided by `dk^(1/4)`
  (equivalent to the standard 1/sqrt(dk) scaling); full softmax over the valid
  positions (self: causal; cross: all encoder frames). Projections with bias.
- **FFN**: `linear2(ReLU(linear1(x)))`, biases present.
- **Flow**: `enc_out [T,512] → enc_dec_proj → [T,1024] → cross-K/V for all
  layers → greedy loop` (beam_size 1 by default in the yaml). The head produces
  5248 logits; argmax; stop at `<|endoftext|>` (id 3) or max_generation_delta
  (50) + prompt length... use max_seq 1024 as a guard.

## Aggregate tokenizer (CanaryTokenizer, type "agg")

Order (= `langs` order in the yaml) and GLOBAL offsets of the SPE BPE sub-tokenizers:

| sub | size | offset |
|---|---|---|
| spl_tokens | 1152 | 0 |
| en | 1024 | 1152 |
| de | 1024 | 2176 |
| es | 1024 | 3200 |
| fr | 1024 | 4224 |

Conversion: tokens.json = flat list of the 5248 pieces in global-id order.
Detokenization identical to the other models (▁ → space); ids < 1152 (specials) are NOT
printed. Key specials (global ids): `<unk>` 0, `<|nospeech|>` 1, `<pad>` 2,
`<|endoftext|>` 3 (EOS), `<|startoftranscript|>` 4, `<|pnc|>` 5 / `<|nopnc|>` 6,
`<|startofcontext|>` 7, `<|itn|>` 8 / `<|noitn|>` 9, `<|timestamp|>` 10 /
`<|notimestamp|>` 11, `<|diarize|>` 12 / `<|nodiarize|>` 13,
`<|emo:undefined|>` 16, languages: `<|en|>` 62, `<|fr|>` 69, `<|de|>` 76, `<|es|>` 169.

## `canary2` prompt format

User template (from `Canary2PromptFormatter`, slots in this order):

```
<|startofcontext|> [decodercontext] <|startoftranscript|> |emotion| |source_lang| |target_lang| |pnc| |itn| |timestamp| |diarize|
```

Defaults from the yaml: decodercontext empty (0 tokens), emotion `<|emo:undefined|>`,
pnc `<|pnc|>`, itn `<|noitn|>`, timestamp `<|notimestamp|>`, diarize `<|nodiarize|>`.
EN ASR prompt (9 tokens): `[7, 4, 16, 62, 62, 5, 9, 11, 13]`.
**Translation** = target_lang ≠ source_lang (e.g. en→de: `[7, 4, 16, 62, 76, 5, 9, 11, 13]`).
The model's response follows the prompt and ends with `<|endoftext|>`.

With `<|timestamp|>` the model emits `<|N|>` tokens (80 ms frames) around the
words — support deferred (v1 engine: notimestamp).

## Validation plan

1. Numpy oracle: reuse the parakeet encoder + new `aed_decode()`; compare
   transcriptions on the fixture WAVs (en + unsupported it → en/de/es/fr) and
   en→de/es/fr translation by inspection.
2. C: per-stage parity vs the oracle (enc_proj, emb, layer 0/N, first-step logits)
   then identical e2e text.
3. Goldens in `make test` with the 180m (downloadable in CI? 735 MB — evaluate).

## The timestamp aligner (canary-1b-v2)

`<|timestamp|>` does not work on canary-1b-v2: it is an AED model, its attention is
not monotonic, and asking for the tokens makes it emit EOS immediately. NVIDIA's own
answer is to ship a **second model** inside the `.nemo` — an alignment ASR whose CTC
posteriors the text is aligned against.

What it actually is, read out of `timestamps_asr_model_config.yaml`:

| | |
|---|---|
| encoder | `ConformerEncoder`, 24 layers, d_model 1024, 8 heads, ffn 4096 |
| subsampling | `dw_striding` 8×, 256 channels, **non-causal** (`att_context_style: regular`) |
| conv | kernel 9, **batch_norm**, `use_bias: false`, `xscaling: false` |
| head | `ConvASRDecoder` → 16384 classes + blank |
| features | 128 mel, n_fft 512, hop 160, **per_feature** normalization |
| weights | 2.5 GB f32, 686 tensors |

That is **the parakeet-ctc family verbatim**, so it converts through the existing
`build_parakeet_ctc_from_yaml` and runs on the existing CTC engine — no new encoder,
no new decoder. `tools/convert_nemo.py` writes it to `<model_dir>/aligner/` with
`"role": "timestamp_aligner"`.

The conversion has an unusually strong test available: the aligner IS an ASR model,
so a correct conversion **transcribes**. It does:

```
$ mynah-asr transcribe -m models/canary-1b-v2/aligner -i tests/audio/test_en.wav
Hello, this is a speech recognition test. The weather is nice today.
```
`tests/test_e2e.sh` asserts exactly that whenever `<model_dir>/aligner/` exists.

### Getting it without downloading 6.4 GB

`canary-1b-v2.nemo` is 6.36 GB and the aligner is 2.5 GB of it. A `.nemo` is an
**uncompressed tar**, so its index can be walked with HTTP range requests — one
512-byte read per member, each header giving the size and therefore the next offset
— and only the wanted members pulled:

| member | size |
|---|---|
| `timestamps_asr_model_weights.ckpt` | 2503.3 MB |
| `timestamps_asr_model_config.yaml` | 0.2 MB |
| `model_weights.ckpt` (the AED model) | 3853.8 MB |

Then `convert_nemo.py --aligner <cfg.yaml> <weights.ckpt> <out_dir>`.

**The trap, paid for once:** the offsets a tar index reports are the offsets of the
*headers*. Member content starts 512 bytes later. Getting that wrong yields a file
shifted by one header, which torch reports as `Cannot use weights_only=True with
files saved in the legacy .tar format` — a message that sends you looking for a
checkpoint-format problem when the real bug is arithmetic. Once the range is right,
the checkpoint is an ordinary zip-format torch file and `weights_only=True` works,
so there is no reason to disable that safety.
