# Parakeet-RNNT 1.1B multilingual (universal tokenizer) - artifact forensics, 2026-10-24

Scope: web/CPU research only. No GPU, no remote boxes, no downloads beyond public metadata JSON, no NGC login, no NIM pull.
Question: is the earlier verdict "weights only on NGC/Riva, so not usable for row transplant" correct?

**Short answer: the verdict was too strong on the technical side and too weak on the legal side.**
- Technically, NGC lists a **`trainable_v1.0`** version (4,011,233,560 bytes). A public third-party repo points to a plain
  **`.nemo`** inside it (`Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo`) and loads it with
  `ASRModel.restore_from`. So a raw NeMo checkpoint very likely exists. Guests cannot download it (an NGC login is needed),
  and we have not inspected it ourselves.
- Legally, the weights are under the **NVIDIA Community Model License**. Copying its rows into our model would make the result a
  **"Derivative Model"** of it. That has consequences we cannot ignore: production use needs NIM/NVAIE, open-source
  relicensing is barred, and the agreement's text has to ship with the model. The tokenizer counts as part of the "NVIDIA Models" too.

## 1. Evidence log (what was actually fetched)

| # | Source | How | Result |
|---|---|---|---|
| E1 | `https://api.ngc.nvidia.com/v2/models/nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal` | anonymous curl, HTTP 200 | `isPublic: true`, `canGuestDownload: false`, `latestVersionIdStr: trainable_v1.0`, `latestVersionSizeInBytes: 4011233560`, `modelFormat: riva`, `precision: fp16` (catalog label), created 2025-03-19, "Model Version(s): Parakeet-RNNT-XXL-1.1b_universal_spe8.5k_1.0", "Riva 2.19.0 or higher", "ready for commercial use", GOVERNING TERMS = NVIDIA Community Model License |
| E2 | same + `/versions`, `/versions/trainable_v1.0`, `/versions/trainable_v1.0/files` | anonymous curl | **HTTP 403 "Artifact is not Guest accessible!"**. No file listing without auth. `/v2/org/nvidia/team/riva/models/...` returns 401 |
| E3 | `https://api.ngc.nvidia.com/v2/search/catalog/resources/MODEL?q=...` | anonymous curl | Sibling entries (table 2). No RMIR entries are searchable. Direct `rmir_asr_parakeet*_unified_ml_cs_universal*` lookups return NOT_FOUND to guests |
| E4 | `https://github.com/Escenda/fluent_audio_ros2/blob/main/scripts/run_live_voice_pipeline_smoke` (line 279) | `gh api` | `asr_model=".../parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo"`. The directory name follows the `ngc registry model download-version` layout (`<name>_v<version>/`) |
| E5 | same repo, `src/ai/fa_asr/fa_asr_py/backends/nemo_parakeet_runtime.py` L160-171 | `gh api` | Loads it with `nemo_asr.models.ASRModel.restore_from(config.model_path)`, uses `model.tokenizer.ids_to_text`, and switches to `greedy_batch` RNNT decoding. This is a working NeMo load path for exactly this file, written by a third party |
| E6 | `https://api.ngc.nvidia.com/v2/repos/nim/nvidia/parakeet-1-1b-rnnt-multilingual` | anonymous curl, 200 | NIM container `nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual`, tags 1.0.0 ... 1.5.0 / latest. amd64 image 13,113,395,635 B compressed, arm64 12,849,781,024 B. Banner: **"End of Support"**. Labels: `Experimental`, `nv-ai-enterprise`, `nim-dev`. Governing terms: NVIDIA Software License Agreement + Product-Specific Terms for AI Products (container) and NVIDIA Community Model License (model). `/images` returns 403 "Not a public artifact" |
| E7 | `https://docs.nvidia.com/nim/speech/latest/asr/deploy-asr-models/parakeet-rnnt.html`, `.../reference/support-matrix/asr.html` | WebFetch | Cache at `/opt/nim/.cache`, export at `/opt/nim/export`. Profiles are selected with `NIM_TAGS_SELECTOR` (`mode=ofl/str/str-thr/all`, `diarizer=sortformer|disabled`, `vad=silero`, `type=default|prompt|indic`). All FP16. NIM "downloads a pre-built model when available for the target GPU or generates an optimized model on-the-fly using the RMIR model". Offline profile ~25.6 GB VRAM, all-modes ~50 GB |
| E8 | `https://github.com/nvidia-riva/tutorials/blob/main/asr-train-and-deploy-NGPU-LM-for-parakeet-rnnt.ipynb` (cells 29-35) | `gh api` | The NIM image itself ships `riva-build`/`riva-deploy` (servicemaker). Example build name `parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-offline` with `--unified_acoustic_model --ms_per_timestep=80`, key `tlt_encode` |
| E9 | `https://github.com/nvidia-riva/nemo2riva` README (MIT licence) | `gh api` | For **Parakeet-RNNT / TDT / Conformer-RNNT / Canary** the documented export is `nemo2riva --key tlt_encode --format nemo`, so the `.riva` wraps the `.nemo` itself inside an EFF archive. Parakeet-CTC exports to ONNX. "`tlt_encode` is the key used for the pre-trained models published on NGC" (Riva docs). EFF library `nvidia-eff` 0.6.7 on PyPI, licence "NVIDIA Proprietary Software" |
| E10 | `https://forums.developer.nvidia.com/t/unable-to-download-rmir-asr-parakeet-1-1b-unified-ml-cs-universal-2-19-0-403-forbidden/344951` | WebFetch | `ngc registry model download-version nvidia/riva/rmir_asr_parakeet_1.1b_unified_ml_cs_universal:2.19.0` returns **403 even with NVAIE keys**. NVIDIA staff replied "you would have to build it". So prebuilt RMIRs are gated. The trainable entry was not discussed |
| E11 | Riva docs `asr-customizing.html` | search snippet | In a deployed Riva model repository, the tokenizer sits in plain form in the decoder directory as `<hash>_tokenizer.model` (general Riva convention, not checked for this model) |
| E12 | `https://huggingface.co/api/models?search=...` (6 queries) | anonymous curl | **No HF mirror or conversion** of the multilingual universal model. Only hit: `kesbeast23/multilingual-parakeet-ctc-1.1b`, which is empty (`.gitattributes` only). Size reference: `nvidia/parakeet-rnnt-1.1b` `parakeet-rnnt-1.1b.nemo` = 4,283,105,280 B (fp32, English, 1024 vocab, CC-BY-4.0) |
| E13 | `https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-community-models-license/` (Last Modified Jan 30 2025, PDF v. Apr 15 2025) | curl + text extract | See section 4 |
| E14 | `build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr/{modelcard,deploy}` | WebFetch | The pages render with JS and the fetch returned only navigation. The "Downloadable" label could not be read from here. E6/E7 cover the same facts |

## 2. Related NGC entries (guest metadata only)

| NGC id (`nvidia/riva/...`) | latest version | bytes | version label / tokenizer | licence on card |
|---|---|---|---|---|
| `parakeet-rnnt-riva-1-1b-unified-ml-cs-universal` | **trainable_v1.0** | 4,011,233,560 | `Parakeet-RNNT-XXL-1.1b_universal_spe8.5k_1.0` | NVIDIA Community Model License |
| `parakeet-ctc-riva-1-1b-unified-ml-cs-universal` | deployable_v1.0 | 3,144,898,117 | `Parakeet-CTC-XXL-1.1b_universal_spe8.5k_1.0` | NVIDIA AI Foundation Models Community License (older PDF) |
| `parakeet-ctc-riva-1-1b-unified-ml-cs-concat` | deployable_v1.0 | 3,154,750,430 | `..._concat_spe12k_1.0` | NVIDIA AI Foundation Models Community License |
| `parakeet-ctc-riva-1-1b-unified-ml-cs-em-ea` | trainable_v1.0 | 7,990,508,592 | EMEA (es, en-GB, de, fr, it, ar) | Riva licence |
| `parakeet-rnnt-riva-1-1b-en-us` | deployable_v8.1 | 3,979,969,823 | English | Riva licence |

Notes:
- Only the latest version id is visible to guests. A `deployable_*` version of the RNNT model may also exist (UNVERIFIED).
- **Size sanity check:** 4.01 GB is about 1.0 B params at fp32, against 4.28 GB for the public fp32 `parakeet-rnnt-1.1b.nemo`.
  An fp16 export would be about 2.1 GB. The size therefore fits a full-precision trainable checkpoint and does not fit an
  fp16 engine. This is an inference from the size, not an observation.
- **Card inconsistency:** the RNNT card says "individual tokenizers ... merged into a single tokenizer", which is the wording of
  the `concat` card. Its version label says `universal_spe8.5k`, and the E4 filename says `merged_universal_spe8.5k`.
  Whether the tokenizer is a single SentencePiece model or a NeMo `AggregateTokenizer` is UNVERIFIED.
- The CTC "universal" sibling has the same `spe8.5k` label. It probably uses the same tokenizer, but that is UNVERIFIED.
- The card also says "8x depthwise-separable convolutional downsampling with **Hybrid loss**". The checkpoint may be an
  `EncDecHybridRNNTCTCBPEModel` with an extra CTC head (UNVERIFIED).

## 3. Verdict per item

| # | Item | Status | Evidence / reasoning |
|---|---|---|---|
| 1 | Model exists / runs locally | **VERIFIED** | NIM image E6 (13.1 GB, tags up to 1.5.0, local docker run documented in E7). Needs `NGC_API_KEY`. Banner says End of Support. Field reports exist (forum threads on streaming confidence, Blackwell profile bugs) |
| 2 | Downloadable NIM/Riva artifact | **VERIFIED** (exists), auth needed | NIM container E6, NGC model `trainable_v1.0` 4.01 GB (E1). Guests are refused (E2). Prebuilt RMIRs are gated even for NVAIE (E10) |
| 3 | Raw NeMo checkpoint (.nemo) | **UNVERIFIED, strong evidence** | Third-party path `..._vtrainable_v1.0/Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo` (E4). The `trainable_*` = .nemo convention appears in Riva tutorials ("You can download the .nemo file ... `speechtotext_zh_cn_conformer:trainable_v5.0`"). Size fits fp32 (sec. 2). We did not see the file listing ourselves. Fallback: if the version holds a `.riva`, then for RNNT it is EFF(`--format nemo`) wrapping the `.nemo` (E9) |
| 4 | Tokenizer accessible | **UNVERIFIED** (very likely, by two routes) | (a) A `.nemo` is a tar that carries `*_tokenizer.model` and `*_vocab.txt` as artifacts. E5 calls `model.tokenizer` successfully. (b) The NIM/riva-deploy model repository holds `<hash>_tokenizer.model` in plain form (E11 convention). Neither route has been inspected for this model |
| 5 | Vocabulary size | **UNVERIFIED** | Labels `spe8.5k` (card) and `merged_universal_spe8.5k` (file name) suggest about 8.5k SentencePiece pieces. The exact count (8,500? 8,192+?) and the blank/special layout are unknown until the tokenizer or config is read |
| 6 | Arabic token ids/pieces inspectable | **UNVERIFIED** | ar-AR is a trained language (5.65 % of validation mix, E1). Pieces become inspectable as soon as item 4 is in hand, with no GPU needed. Today: not inspected |
| 7 | Prediction network weights | **UNVERIFIED, strong evidence** | Present in any trainable RNNT `.nemo` (`decoder.prediction.*`), and E5 decodes with them. Not seen by us |
| 8 | Joint network weights | **UNVERIFIED, strong evidence** | Same as 7 (`joint.*`) |
| 9 | Encoder weights | **UNVERIFIED, strong evidence** | Same as 7 (`encoder.*`, about 1.0 B params by size) |
| 10 | NeMo loadability | **UNVERIFIED, strong evidence** | E5: `ASRModel.restore_from` on this exact file in a public repo. NeMo version needed is unknown (Riva 2.19-era, about NeMo 2.2+). If it is hybrid, it may need `EncDecHybridRNNTCTCBPEModel` |
| 11 | Legal reusability for our research fine-tune | **VERIFIED: restricted** | See section 4. Research use and Derivative Models are allowed. A row transplant turns our model into a Derivative Model under this license. The NIM-container route (TRT engines) and decrypting RMIRs are worse |

Earlier claim "weights only on NGC/Riva" means "auth-gated NGC download". It does not mean "unavailable". The
**NOT-AVAILABLE** label applies to only two things: (a) guest/anonymous download and (b) any Hugging Face mirror.

## 4. Licence analysis (NVIDIA Community Model License, v. 2025-01-30 / PDF 2025-04-15)

The model is governed by this license (E1, E6). Its definition of "NVIDIA Models" explicitly covers
"software, checkpoints, learnt weights, algorithms, parameters, **configuration files**...", so the tokenizer and config are covered too.

| Clause | Text (abridged) | Impact on us |
|---|---|---|
| 1.2 General Downloadable Grant | may install/use/reproduce, **create Derivative Models**, distribute with a copy of the Agreement, "except that to exercise these grants **in production** you must use NVIDIA NIM ... through an NVIDIA AI Enterprise subscription", unless an RTX-local or listed end-device exemption applies | Offline research and evaluation are allowed. Serving a derived model in production outside NIM/NVAIE is not |
| 1.4 Derivative Models | includes "any other model that is created or **initialized by transfer of patterns of the weights, parameters** ... to the other model", including distillation | **Copying the embedding or joint rows into our model makes our model a Derivative Model.** Initialising from them counts too |
| 2.1.2 | may not use NVIDIA Models, Derivative Models or outputs "to develop or improve any other AI models (excluding NVIDIA Models or Derivative Models)" without written approval (email is enough) | Pseudo-labelling or distilling into a model that would stay "ours" is barred. A transplant is "permitted" only because the result becomes a Derivative Model |
| 2.1.4 | may not use NVIDIA Models "in any manner that would cause them to become subject to an open source software license" | We cannot publish a transplanted model under CC-BY/Apache/MIT. Releasing it on the Hub requires this agreement instead |
| 2.1.6 | may not "bypass ... or circumvent any ... **encryption**, security, DRM, or authentication mechanism" | Decrypting an RMIR/.riva outside Riva tooling, even with the documented `tlt_encode` key, or pulling weights out of NIM TRT engines, is a legal grey zone. **Only the plain trainable `.nemo` route is clean** |
| 2.2.1 | modified files must carry prominent change notices | Required on any derived checkpoint |
| 6 | automatic termination on non-compliance; destroy all copies | Risk stays with whoever holds the derived model |
| Container | NIM image under NVIDIA SLA + Product-Specific Terms for AI Products | Extracting assets from the container is a separate, stricter regime. Avoid |

Practical reading (this is not legal advice):
- **Allowed:** download the trainable checkpoint with a personal NGC key, inspect the tokenizer and Arabic pieces, run research
  fine-tunes of *this* model, and evaluate.
- **Allowed, but with consequences:** transplanting rows into our CC-BY-based model. The result is a Derivative Model. It is then
  non-open-source, ships with this agreement, and can go to production only through NIM/NVAIE.
- **Not allowed without NVIDIA written approval:** using its outputs (pseudo-labels) or distillation to improve a model we
  want to keep outside this license.
- **Using only the tokenizer** (inventory and segmentation design ideas) is the lowest-risk use. Copying the `.model` file is
  still copying an "NVIDIA Model" component.

## 5. Exact access steps (not performed)

Needs a free NVIDIA developer / NGC account and a **personal API key**. Whether a free key is enough for this entry is UNVERIFIED.
E10 shows that some `nvidia/riva` artifacts 403 even for NVAIE keys. Run the steps on a box, not on the Mac (download about 4.0 GB).

1. Metadata-only check, zero cost, decisive:
   ```bash
   ngc config set            # paste personal key, org=nvidia
   ngc registry model info nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal          # all versions
   ngc registry model info nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0 --files
   # or REST:
   TOKEN=$(curl -s -u "\$oauthtoken:$NGC_API_KEY" \
     "https://authn.nvidia.com/token?service=ngc&scope=group/ngc:nvidia&group/ngc:nvidia/riva" | jq -r .token)
   curl -s -H "Authorization: Bearer $TOKEN" \
     "https://api.ngc.nvidia.com/v2/org/nvidia/team/riva/models/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal/versions/trainable_v1.0/files"
   ```
   Expected inventory: one file `Parakeet-RNNT-XXL-1.1b_merged_universal_spe8.5k_1.0.nemo` of about 4.01 GB, possibly
   plus a README. If the response is 403, record "auth-gated beyond free tier" and stop.
2. Download: `ngc registry model download-version nvidia/riva/parakeet-rnnt-riva-1-1b-unified-ml-cs-universal:trainable_v1.0`
   gives `./parakeet-rnnt-riva-1-1b-unified-ml-cs-universal_vtrainable_v1.0/*.nemo` (about 4.0 GB, one file).
3. CPU inspection, no weights loaded:
   ```bash
   tar tvf X.nemo                                   # expect model_config.yaml, model_weights.ckpt, *_tokenizer.model, *_vocab.txt (+ tokenizer.vocab)
   tar xf X.nemo --wildcards '*.yaml' '*tokenizer*' '*vocab*'
   python -c "import sentencepiece as s;m=s.SentencePieceProcessor(model_file='..._tokenizer.model');print(m.vocab_size())"
   # grep model_config.yaml: target, tokenizer.type (bpe vs agg), decoder.vocab_size, prednet.pred_hidden, joint.jointnet.joint_hidden, encoder.n_layers/d_model
   # Arabic: list pieces in U+0600-06FF; encode a FLEURS-ar sample, count <unk>
   ```
4. Tensor shapes only, CPU: `torch.load(model_weights.ckpt, map_location='cpu', mmap=True)`, then print the shapes of
   `decoder.prediction.embed.weight`, `joint.joint_net.*`, `joint.pred`, `joint.enc` (with `mmap` this needs little RAM).
5. Load check (box): `ASRModel.restore_from(X.nemo, map_location='cpu')` on current NeMo, then transcribe one Arabic clip.

If the version turns out to hold a `.riva` instead: it is an EFF archive. For RNNT it wraps the `.nemo` (`--format nemo`), and the
public key is `tlt_encode`. Opening it outside Riva tooling falls under the clause 2.1.6 grey zone. Ask before doing it.

NIM route, not recommended for weights: `docker pull` 13.1 GB, then `nim_list_model_profiles` and `nim_download_to_cache`
fill `/opt/nim/.cache` with prebuilt Triton/TensorRT model repositories (engines plus a plain `<hash>_tokenizer.model` by Riva
convention). This route gives at most the tokenizer. Weights inside TRT plans are not usable for transplant and are
licence-encumbered.

## Status: NEXT-AGENT

Next concrete step: **user decision first, then step 5.1 (metadata-only, authenticated file listing).** It needs a personal NGC
API key on a box and costs nothing in download or GPU. It turns item 3 from UNVERIFIED to VERIFIED or NOT-AVAILABLE. If item 3
is verified, run 5.2 and 5.3 on CPU to settle items 4-6 (vocab size, Arabic pieces). Before any row transplant, the user decides
whether a Community-Model-License derivative is acceptable for the Arabic model's intended release. If it is not, use the
tokenizer for inventory only and keep the transplant on CC-BY sources.
