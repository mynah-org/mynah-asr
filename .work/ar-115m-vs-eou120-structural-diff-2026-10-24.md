# NVIDIA Arabic FastConformer 115M vs EOU-120M: tensor-level structural diff, and arm T3 (2026-10-24)

**Status: GPU-READY.** The T3 checkpoint is built and passes the CPU structural gate (19 of 19 checks). AR-A0
amendment 1 (T3 frozen + the pending T unfrozen diagnostic) is pre-registered, committed and shipped to dgx2. It has
not been launched.

Every number below was OBSERVED on dgx1, on CPU only (`nice -n 19`, 2 threads): the configs and state_dicts were read
straight from the two `.nemo` tarballs. Points marked HYPOTHESIS are interpretation.

- Code: `finetune/eou/diag/ar_t3_struct.py` (subcommands `matrix`, `build`, `test`, `sanity`),
  `finetune/jobs/ar_a0_t3.sh`, `ar_a0_score.py --table`, and amendment 1 in `plans/AR-A0-prereg.md`.
- Commit `51d7e027` on worktree `mynah-asr-wt-trainB`, branch `feat/trainer-throughput`. Local only, not pushed.
  It reached both boxes as a bundle: dgx1 `~/ft/art3.bundle`, dgx2 `~/ft/art3.bundle`, plus a worktree `~/art3-wt`
  on dgx2.
- Artefacts:
  - dgx1 `~/tmp/art3/out/{matrix.json,tensors.tsv}`: the per-tensor table, 707 + 467 names.
  - `~/ft/models/ar-t3-candidate/` on dgx1 and dgx2: `ar_t3.nemo` (sha256 `3383f973…f83c0`), `build.json`,
    `struct_test.json`, `tokenizer/`, `joint_enc_bridge.pt`.
- Models:
  - `nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0` = `~/ft/models/stt_ar_fc_hybrid_large_pc_v1.0.nemo`, CC-BY-4.0,
    114.66 M params, NeMo 2.0.0rc1, `EncDecHybridRNNTCTCBPEModel`.
  - EOU-120M = `parakeet_realtime_eou_120m-v1.nemo`, 114.89 M params, NeMo 2.6.0rc0, `EncDecRNNTBPEModel`.

## 1. Config matrix (only the differing keys are flagged; every key read from `model_config.yaml`)
| key | EOU-120M | AR 115M | same? |
|---|---|---|---|
| encoder n_layers / d_model / n_heads / ff_expansion | 17 / 512 / 8 / 4 | 17 / 512 / 8 / 4 | yes |
| conv_kernel_size | 9 | 9 | yes |
| **conv_norm_type** | layer_norm | batch_norm | **no** |
| **conv_context_size** | causal (left 8, right 0) | unset = symmetric (4 / 4) | **no** |
| **att_context_size / style** | [70, 1] chunked_limited | [-1, -1] regular (full context) | **no** |
| **causal_downsampling** | true (CausalConv2D) | unset = false | **no** |
| self_attention_model / pos_emb_max_len | rel_pos / 5000 | rel_pos / 5000 | yes |
| **xscaling** | false | true (x * sqrt(512) after pre-encode) | **no** |
| **use_bias** | false (no bias in any MHSA / FFN / conv linear) | unset = true | **no** |
| subsampling / factor / channels | dw_striding / 8 / 256 | dw_striding / 8 / 256 | yes |
| **feat_in** | 128 | 80 | **no** |
| dropout / pre_encoder / emb / att | 0.1 / 0.1 / 0.0 / 0.1 | same | yes |
| **preprocessor features / normalize** | 128 / NA | 80 / per_feature | **no** |
| preprocessor window / size / stride / n_fft / dither | hann / 0.025 / 0.01 / 512 / 1e-5 | same | yes |
| decoder: pred_hidden / rnn layers / blank_as_pad / dropout | 640 / 1 / true / 0.2 | same | yes |
| **decoder.vocab_size** | 1026 | 1024 | **no** (vocab) |
| joint: enc 512, pred 640, joint 640, relu, dropout 0.2, fused bs 4 | same | same | yes |
| **joint.num_classes** | 1026 | 1024 | **no** (vocab) |
| **aux CTC** | absent | ConvASRDecoder 512 -> 1025, weight 0.3 | **no** |
| **loss fastemit_lambda** | 0.03 | 0.0 | no (our stage 1 uses 0 anyway) |
| **spec_augment freq / time masks** | 2 / 10 | 4 / 20 | no (training-time only) |

Tokenizers (read from the SentencePiece protos):
| | EOU-120M | AR 115M |
|---|---|---|
| SP model type | BPE (2) | **UNIGRAM (1)** |
| pieces | 1026 = 1024 + `<EOU>` 1024 + `<EOB>` 1025 (USER_DEFINED) | 1024, no user-defined pieces |
| unk / bos / eos / pad | 0 / -1 / -1 / -1 | 0 / -1 / -1 / -1 |
| normalizer | nmt_nfkc_cf (case fold) | nmt_nfkc |
| byte_fallback / coverage | false / 1.0 | false / 1.0 |
| RNNT blank | 1026 | 1024 |
| content | English, lower case | Arabic, P&C model: `،` 21, `.` 31, `؟` 109; 99 of 1024 pieces contain hamza-alef forms (أ إ آ ٱ); 0 tashkeel pieces |

## 2. Tensor matrix (every tensor; the 17 encoder layers collapse to one row each, and the code asserts that every row is uniform across layers)
Totals: 462 EXACT (111.32 M params), 3 ROW-MAP, 2 IMPOSSIBLE, 240 ABSENT-EOU (0.70 M AR params: biases and BN
statistics), 0 ABSENT-AR.

"EXACT" means the name and shape match, nothing more. The semantics column decides whether a copy would mean anything.

| tensor (x17 for `layers.{i}`) | EOU | AR | class | semantic note |
|---|---|---|---|---|
| preprocessor.featurizer.fb | [1,128,257] | [1,80,257] | IMPOSSIBLE | 128 vs 80 mel bins; plus NA vs per_feature normalisation |
| preprocessor.featurizer.window | [400] | [400] | EXACT | hann buffer, identical function |
| encoder.pre_encode.conv.{0,2,3,5,6}.{weight,bias} | 3x3 dw / 1x1 pw, 256 ch | same | EXACT (10) | EOU CausalConv2D (left/top pad) vs symmetric pad: same kernels, shifted receptive field |
| encoder.pre_encode.out.weight | [512, 4352] | [512, 2560] | IMPOSSIBLE | 256 x freq-bins: 128 causal -> 17, 80 non-causal -> 10 (checked: 80 causal -> 2816, 128 non-causal -> 4096) |
| encoder.pre_encode.out.bias | [512] | [512] | EXACT | the input scale differs: AR multiplies by sqrt(512) (xscaling) before layer 0 |
| layers.{i}.feed_forward{1,2}.linear{1,2}.weight | 2048x512 / 512x2048 | same | EXACT | AR also has biases |
| layers.{i}.feed_forward{1,2}.linear{1,2}.bias | - | [2048]/[512] | ABSENT-EOU | use_bias=false in EOU |
| layers.{i}.norm_{feed_forward1,self_att,conv,feed_forward2,out}.{weight,bias} | [512] | [512] | EXACT | LayerNorms, same op |
| layers.{i}.self_attn.linear_{q,k,v,out}.weight | 512x512 | same | EXACT | EOU masked to [70 left, 1 right] chunked_limited; AR full context |
| layers.{i}.self_attn.linear_{q,k,v,out}.bias | - | [512] | ABSENT-EOU | use_bias |
| layers.{i}.self_attn.linear_pos.weight, pos_bias_u, pos_bias_v | 512x512, [8,64] x2 | same | EXACT | rel-pos, different context window |
| layers.{i}.conv.pointwise_conv1.weight / pointwise_conv2.weight | [1024,512,1] / [512,512,1] | same | EXACT | AR also has biases |
| layers.{i}.conv.pointwise_conv{1,2}.bias | - | [1024]/[512] | ABSENT-EOU | use_bias |
| layers.{i}.conv.depthwise_conv.weight | [512,1,9] | [512,1,9] | EXACT | **same shape, different op**: EOU causal (taps t-8..t), AR centred (t-4..t+4). A copy would shift every kernel by 4 frames (320 ms) |
| layers.{i}.conv.depthwise_conv.bias | - | [512] | ABSENT-EOU | use_bias |
| layers.{i}.conv.batch_norm.{weight,bias} | [512] | [512] | EXACT | **same name and shape, different op**: in EOU the module is a LayerNorm (the attribute keeps the name `batch_norm`), in AR a BatchNorm1d |
| layers.{i}.conv.batch_norm.{running_mean,running_var,num_batches_tracked} | - | [512],[512],[] | ABSENT-EOU | BN statistics |
| decoder.prediction.embed.weight | [1027,640] | [1025,640] | ROW-MAP | blank_as_pad: the last row (blank) is the zero pad row on both sides |
| decoder.prediction.dec_rnn.lstm.{weight_ih,weight_hh,bias_ih,bias_hh}_l0 | 2560x640 x2, [2560] x2 | same | EXACT | 1-layer LSTM 640; each side models its own token histories |
| joint.pred.{weight,bias} | 640x640, [640] | same | EXACT | consistent only with the predictor it was trained with |
| joint.enc.{weight,bias} | 640x512, [640] | same | EXACT | **shape-compatible, semantically incompatible**: it reads its own encoder's feature basis (measured in section 3) |
| joint.joint_net.2.{weight,bias} | [1027,640], [1027] | [1025,640], [1025] | ROW-MAP | rows = token ids, last = blank |
| ctc_decoder.decoder_layers.0.{weight,bias} | - | [1025,512,1], [1025] | ABSENT-EOU | aux CTC head (AR only) |

### Conclusion, tensor by tensor
- **Encoder (all 442 EXACT encoder tensors): no transfer is meaningful in either direction.** The shapes match, but:
  - every conv-module norm is a different op (LN vs BN);
  - every depthwise kernel has a different tap alignment;
  - every linear lacks the AR bias it was trained with;
  - the residual stream has a different scale (xscaling);
  - the input front-end is IMPOSSIBLE (pre_encode.out, fb).

  Matching shapes are not compatibility here. In line with the user decision, no encoder tensor is transplanted.
- **Text side: semantically compatible as one block.** The AR decoder embedding, LSTM, `joint.pred` and output rows
  form the AR predictor -> joint-hidden -> token chain. None of these tensors touches the encoder, so the block can be
  moved whole, with an exact row map.
- **`joint.enc`: not compatible**, although its shape is EXACT. It is the only tensor at the encoder/text interface.

## 3. Is copying `joint.enc` meaningful? Measured, not assumed
`ar_t3_struct.py build` paired the EOU-120M encoder frames with the AR model's frames on the same audio: 400 CV17-ar
train clips, 15.7 k train frames and 4.0 k held-out frames; the lengths differ by 0 or 1 frame. The target was the AR
`joint.enc` output.

- Raw AR `joint.enc` applied to EOU features: held-out R^2 = **-6.5**. It is worse than predicting the mean, which
  means the feature bases are unrelated.
- Best linear (ridge) bridge from EOU frames to the AR `joint.enc` output: held-out **R^2 = 0.129** at lag 7 frames
  (EOU lagging, 560 ms) with lambda 0.1. At lag 0 it is 0.058, and R^2 peaks smoothly between lags 6 and 7.
  - HYPOTHESIS: the lag is the streaming encoder (1-frame right context) committing later than the full-context one.
  - Even so, 87 % of the variance is not linearly recoverable: the two encoders encode Arabic speech differently,
    beyond a rotation.
- Zero-shot on 50 CV17-ar test clips (not the microset):

| model | greedy WER / CER | RNNT nats / token | output |
|---|---|---|---|
| AR 115M itself (full context) | 5.84 / 1.78 | 1.88 | correct, with punctuation |
| T3, AR `joint.enc` copied | 100 / 100 | 44.10 | all empty |
| T3, ridge bridge | 99.64 / 99.38 | 7.39 | almost all empty |
| **T3, EOU-120M's own `joint.enc`** (chosen) | 100 / 100 | **6.06** | all empty |
| T control (same tokenizer, EOU text side with token rows re-drawn = arm T) | 5282 / 8298 | 711.5 | babble (the AR-A0 T failure mode) |

Verdict: copying the AR `joint.enc` is not meaningful. Its loss is 7x worse than EOU's own projection.

The pre-registered rule picked among the non-transplanted options by the lower zero-shot loss. The winner is
EOU-120M's own `joint.enc` (6.06 vs 7.39 for the bridge), and that is what `ar_t3.nemo` carries. The bridge weights
are kept in `joint_enc_bridge.pt`.

T3 starts as "silent" (blank-dominant), never as babble. On the same audio and targets its loss at step 0 is 117x
lower than arm T's. HYPOTHESIS: the micro-overfit then only has to train the 512 -> 640 interface (plus fine-tune) on a
coherent pretrained Arabic predictor. Arm T, by contrast, must unlearn random rows.

## 4. T3 as built (`~/ft/models/ar-t3-candidate/ar_t3.nemo`, sha256 `3383f97301e7827747dc9dbfee9ce3220a23766d4ffe83931b7bb00e186f83c0`)
- Encoder and preprocessor: the EOU-120M ones, **byte-identical**, with the encoder and preprocessor config identical
  (cache-aware streaming cfg: chunk [9,16], last_channel_cache 70, pre_encode_cache [0,9]).
- Tokenizer: the AR SentencePiece with `<EOU>` and `<EOB>` appended as USER_DEFINED pieces (`tokenizer_eou.add_special`).
  Vocab 1026, `<EOU>` 1024, `<EOB>` 1025, blank 1026, `<unk>` 0. Pieces 0..1023 are identical to the AR tokenizer.
- Row map for `decoder.prediction.embed.weight`, `joint.joint_net.2.weight` and `joint.joint_net.2.bias`:
  - new[0:1024] = ar[0:1024];
  - new[1026] (blank) = ar[1024] (blank; the embedding row is zero);
  - new[1024] `<EOU>` = **explicitly initialised** as the mean of the AR rows of `.` (31) and `؟` (109), which are the
    AR model's own end-of-sentence pieces and live in the same joint-hidden basis;
  - new[1025] `<EOB>` = **preserved** from EOU-120M row 1025.
  - HYPOTHESIS: an EOU-120M row in a foreign basis is close to a random direction with a trained bias. Stage 1 has no
    `<EOU>` targets, so these rows matter only for the later EOU stage.
- Copied verbatim from AR: the prediction LSTM (4 tensors) and `joint.pred` (2).
- `joint.enc` = EOU-120M's own (section 3).
- CPU structural gate (`struct_test.json`): **PASS, 19 of 19 checks.**
  - vocab, ids and blank: vocab 1026; `<EOU>`/`<EOB>` at 1024/1025; blank 1026 in the decoder, the joint (1027
    outputs) and the greedy decoder.
  - tokenizer: pieces identical to AR; the probe `هل تلعب كرة القدم<EOU>` gives the same ids as the AR tokenizer,
    plus a single 1024.
  - tensors: names and shapes equal to EOU-120M; every non-text-side tensor byte-identical to EOU-120M; text-side rows
    byte-identical to AR per the row map; the blank embedding row is zero; `joint.enc` is not the AR copy.
  - configs: encoder, preprocessor and streaming cfg unchanged.
  - runtime: forward + RNNT loss (67.5) + backward on 2 Arabic clips, all finite; one `cache_aware_stream_step` runs.

## 5. Reverse option: start from AR 115M and make it streaming + EOU (not recommended)
These are the changes needed, with each one's consequence.

- **Configs:**
  - `att_context_size [70,1]` + `chunked_limited`: weights reusable, but every layer must re-learn under the mask.
  - `conv_context_size causal`: the 17 depthwise kernels re-align; the effective re-init is about 4 taps.
  - `causal_downsampling true`: `pre_encode.out` 2560 -> 2816 becomes IMPOSSIBLE (re-init), and the conv kernels
    re-align.
  - **`normalize per_feature -> NA`**: per-utterance feature normalisation cannot run in a stream. The whole input
    distribution changes.
  - Optional: LN instead of BN in the conv module (NVIDIA's streaming recipe; it would be 17 more re-inits).
- **Vocab:** `<EOU>`/`<EOB>` added, blank 1024 -> 1026. The embedding, joint output and CTC head become ROW-MAP
  1025 -> 1027, and the blank row moves.
- **Training needed:** EOU-120M was itself trained from scratch as a streaming model. Converting an offline encoder is
  a full re-training of the acoustic side on a large Arabic set, followed by the whole EOU stage.
- HYPOTHESIS: this is far more GPU than AR-A1 and outside the frozen recipe. The only part it keeps intact is the text
  side, which T3 already reuses.

## 6. The three arms
| arm | tokenizer / targets | text side at init | inherits | AR-A0 result |
|---|---|---|---|---|
| R | stock English BPE, reversible romanisation | EOU-120M stock (English LM over [a-z']) | EOU-120M acoustic encoder + English predictor; nothing Arabic | **MEMORISE** (CER 32.91 @100, 0.00 from @250) |
| T | new SP-1024 trained on CV/FLEURS text | token rows re-drawn; LSTM and joint projections English | encoder + joint projections; no token knowledge | **FAIL**, babble (CER 4423 @100 -> 616 @1000); zero-shot loss 711 nats/token |
| T3 | NVIDIA Arabic SP-1024 (+`<EOU>`/`<EOB>`) | AR 115M pretrained predictor, joint.pred and output rows; EOU's own joint.enc | encoder (EOU) + an Arabic token LM / joint-output head trained on NVIDIA's Arabic corpus; missing the encoder interface | pending (GPU-READY); zero-shot: silent, 6.06 nats/token |

Known handicap of T3 under the frozen AR-A0 normaliser: `norm_ar` folds hamza forms to bare alef, so 99 of the 1024
pieces (9.7 %) can never be targets. This is kept for comparability with R/T. A later hamza-preserving target set
would need its own pre-registration.

## 7. GPU jobs (GPU-READY on dgx2; not launched)
`finetune/jobs/ar_a0_t3.sh` in dgx2 `~/art3-wt` (commit 51d7e027). It works in this order:

1. Verifies the T3 sha256.
2. Runs the step-0 Arabic decode sanity on CPU (nice 19, 2 threads).
3. **Waits** on `~/gpu.lock` (flock, blocking: it queues behind the serving queue).
4. Runs the two training jobs:
   - T3 frozen (`--stock ar_t3.nemo --freeze-enc 1`);
   - T unfrozen (`--spm ar_a0_spm --freeze-enc 0`).

   Both use the same microset, 1000 steps, lr 3e-4, bs 16, seed 0 and eval steps 100/250/500/750/1000; greedy only
   (`--beam-size 0`); `timeout 1500` each.
5. Releases the lock, scores R/T/T3/T-unfrozen into `results/ar_a0/score_t3.json`, and writes the per-step table to
   `results/ar_a0/convergence_t3.md`.
6. Runs a final Arabic decode sanity on the trained T3.

Both jobs have passed `--dry-run` on dgx2 (32 utts, eval steps resolved).

Launch on dgx2:
```
cd ~/art3-wt && mkdir -p ~/ft/logs && FT_ROOT=$HOME/ft VENV=$HOME/nemo-venv GPU_LOCK=$HOME/gpu.lock \
  tmux new -d -s ara0t3 'bash finetune/jobs/ar_a0_t3.sh 2>&1 | tee -a $HOME/ft/logs/ar_a0_t3.log'
```
Expected wall time once the lock is held: about 4 min for T3 (T frozen took 163 s without beam) and about 4-5 min for
T unfrozen, plus 2 x ~1 min of CPU sanity.

Decision rules are in amendment 1 of `plans/AR-A0-prereg.md`. For example, T3 MEMORISE with CER@100 < 32.91 means a
T3 arm is added to AR-A1.

## NEXT-AGENT
- Launch the command above on dgx2 (it waits for the lock by itself).
- Paste `convergence_t3.md` and both sanity JSONs here.
- Apply the amendment-1 table.
