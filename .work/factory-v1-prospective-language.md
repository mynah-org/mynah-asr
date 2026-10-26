# Factory v1: which new language for the prospective frozen-recipe validation (2026-10-26)

CPU-only desk research: metadata only, no audio downloaded, no GPU used. Companion to `factory-validation-matrix.md`
and `eou-specialist-evolution.md`. Tags: **OBS** = observed, with a source; **HYP** = hypothesis or estimate;
**UNK** = unknown. "Loader" means the existing code in `mynah-asr-wt-trainB/finetune/`:
- `canary/prepare_it.py`: MLS, plus the FLEURS eval;
- `eou/prepare_mix.py`: CV17 via `fixie-ai/common_voice_17_0`, VP via `facebook/voxpopuli`, FLEURS train, and
  Granary-YODAS via `espnet/yodas-granary`;
- `jobs/granary_build.sh` / `jobs/a2_vpfilter.sh`: the Nemotron teacher;
- `eou/prepare_final_it.py`: the FINAL banks.

Sources used below:
- [HF-DS] HF datasets-server `/splits` and `/size`, queried 2026-10-26: row counts and parquet GB per config;
- [HF-TREE] the HF tree API for `espnet/yodas-granary/data/*`;
- [MLS] arXiv 2012.03411, Table 2;
- [VP] github.com/facebookresearch/voxpopuli README;
- [NEMO] huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b, plus `mynah-asr/docs/nemotron-languages.md`;
- [CVPT] MuPe paper (aclanthology 2025.coling-main.407), which quotes the CV17-pt statistics;
- [TN] a local run of `common/textnorm.py`.

## 1. Short list and verdicts

| rank | language | verdict (one line) |
|---|---|---|
| **1** | **Dutch (nl)** | The only candidate that fills **every clause of the factory-v1 data contract unchanged**: MLS, CV17, VP, FLEURS and Granary-YODAS all exist, the teacher is in Nemotron's top tier, and the text rule needs nothing beyond the ES map. It sits outside Romance, so it is a real out-of-sample test, but it keeps the Latin script and the scope stays honest |
| 2 | Portuguese (pt) | The natural "4th Romance" language with the richest eval side (mTEDx-pt, Speech-MASSIVE pt-PT). But **VoxPopuli has no transcribed pt**, so the frozen 4-source contract cannot be met unchanged. The BR / PT split is a confound, and pt is the closest to ES, so it is the weakest generalisation test |
| 3 | Polish (pl) | Every source exists. But **MLS-pl has only 11 train speakers and 4 + 4 dev / test speakers**, `ł` breaks `text_ok` (a new pre-map is needed), the teacher is weaker (FLEURS 15.15 %), and it is Slavic with heavy inflection. It tests "a hard language", not "the factory transfers" |

Considered and dropped:

| language | why it was dropped (evidence) |
|---|---|
| Romanian | **No MLS config** (OBS [HF-DS]: MLS has only dutch / german / polish / portuguese among the candidates). CV17-ro train has only 5,141 clips (~7 h, HYP at ~5 s per clip). Granary-ro is 12,276 samples / 3.7 GB (OBS, the yodas-granary card through search; `data/ro000` = 4.4 GB [HF-TREE]). Teacher FLEURS WER 25.90 % (OBS [NEMO]). It would fail the human-source contract twice |
| Catalan | No MLS and no VP. **No Catalan in Granary or in Nemotron's 40 locales, so there is no teacher filter.** It is absent from the fixie CV17 mirror (OBS: 50 configs, no `ca`), so it would need a new MDC (API key) loader. Only FLEURS `ca_es` exists |
| Galician | Not in MLS / VP / Granary / Nemotron. Only CV17-gl (25,159 train rows, OBS [HF-DS]) and FLEURS `gl_es`. One source |
| German | Viable (MLS 1,966 h / 176 speakers, VP 282 h, CV huge, Tier-1 teacher), but: **`ß` fails `text_ok`** (OBS [TN]: "straße" -> "stra e", text_ok False), so it needs a new pre-map; the downloads are about 2x Dutch (MLS train 30.9 GB, VP train 50.8 GB, OBS [HF-DS]); compounding makes WER noisy. It is the Germanic backup if Dutch is vetoed |

## 2. Data availability per source (train side; caps = MLS 40 h over all speakers, CV <= 20 per speaker, VP t50 at <= 40 h, FLEURS train)

| source | Dutch (nl) | Portuguese (pt) | Polish (pl) |
|---|---|---|---|
| MLS train h / speakers | 1,554.2 h / 40 (9 M, 31 F) OBS [MLS]; 48 shards, 23.8 GB (OBS [HF-TREE]) | 160.96 h / 42 (26 M, 16 F) OBS [MLS]; 6 shards, 2.5 GB | 103.65 h / **11** (6 M, 5 F) OBS [MLS]; 4 shards, 1.6 GB |
| MLS dev / test (h, speakers) | 12.76 / 12.76 h; 6 / 6 speakers; 3,095 / 3,075 rows (OBS) | 3.64 / 3.74 h; 10 / 10; 826 / 871 rows (OBS) | 2.08 / 2.14 h; **4 / 4**; 512 / 520 rows (OBS) |
| -> obtainable under the cap | 40 h (HYP: certain, 1,554 h pool) | 40 h (HYP) | 40 h, but from 11 voices (HYP) |
| CV17 (fixie mirror) train rows | 34,898 (validated 90,449) OBS [HF-DS] | 21,968 OBS [HF-DS]; CV17-pt overall 211 h, 175 h validated, 3,453 speakers (OBS [CVPT]) | 20,729 (validated 132,661) OBS [HF-DS] |
| -> CV hours under <= 20 / speaker | ~35-40 h (HYP: 34,898 x ~4-5 s = 39-48 h before the cap) | **~22-28 h** (HYP: below the 40 h cap, because the loader reads `train` only) | ~23-28 h (HYP; `validated` is larger, but using it would be a loader change) |
| VoxPopuli transcribed | 53 h / 221 speakers (OBS [VP]); train 20,968 rows, 8.83 GB (OBS [HF-DS]) | **none** (OBS [VP]: "—" for pt; no `pt` config in `facebook/voxpopuli`) | 111 h / 282 speakers (OBS [VP]); train 34,665 rows, 18.7 GB |
| -> VP t50 under the cap | ~35-40 h (HYP: train ~45 h; ES / FR / IT t50 kept 80-99 %) | 0 | 40 h (HYP) |
| FLEURS train | `nl_nl` 2,918 rows (OBS) -> ~7-9 h (HYP) | `pt_br` 2,793 rows (OBS) -> ~7-9 h; **BR only** | `pl_pl` 2,841 rows (OBS) -> ~7-9 h |
| Granary-YODAS (`espnet/yodas-granary`, already the IT/FR loader) | `nl000` + `nl100`: 630 files, 155 GB; ~866k samples (OBS [HF-TREE] + card) | `pt000`, `pt100`-`pt103`: 4,162 files, ~1.5 TB; ~5.9M samples (OBS) | `pl000`: 280 files, 78 GB; ~264k samples (OBS) |
| teacher (Nemotron 0.6B), FLEURS WER | nl-NL **11.46 %**, Tier 1 (OBS [NEMO]) | pt **5.48 %**, Tier 1; separate pt-BR / pt-PT prompts (OBS [NEMO]) | pl-PL 15.15 % (OBS [NEMO]); the local doc lists pl as Tier 2 (OBS, docs/nemotron-languages.md) |
| human 4-source total (HYP) | **~120-127 h, the same shape as ES 126.5 / FR 127.4** | ~70-77 h (3 sources), i.e. a contract deviation | ~110-117 h, with MLS being 11 voices |

## 3. Script / tokenizer, diversity, eval, leakage, difficulty, licence

| axis | Dutch | Portuguese | Polish |
|---|---|---|---|
| **stock-SPE representability** (the stock SPE maps accented vowels to `<unk>`, so we de-accent: evolution doc l.62) | é / ë / ï / ĳ: NFKD -> a-z, and `text_ok` passes (OBS [TN]: "zo'n ĳs één ruïne" -> "zo'n ijs een ruine"). Apostrophe words ('s, z'n, m'n) survive in `[a-z' ]`. **Needs only `PRE["nl"] = _ES`** (a typographic-apostrophe map) | ã õ ç á â ê: NFKD -> a-z (OBS [TN]: "ação coração pão" -> "acao coracao pao"). `PRE["pt"] = _ES` | **`ł` does not decompose under NFKD.** Every clip containing it would be dropped by `text_ok` (OBS [TN]: "łódź ..." -> False; "zdzb o"). Needs a new `ł -> l` pre-map, and de-accenting merges many minimal pairs (ą / a, ż / z) |
| homographs created by de-accenting | few: Dutch rarely uses diacritics (HYP) | some (avó / avô, é / e), as in IT / ES (HYP) | many (HYP) |
| speakers and geography | MLS 40 voices; CV and VP have many; nl-NL vs Flemish (nl-BE) is mixed in CV / VP / MLS, but the orthography is identical: a mild confound (HYP) | **BR vs PT is a real confound**: FLEURS is BR only, Speech-MASSIVE pt-PT is PT only, the MLS / CV / Granary splits are UNK (HYP: mostly BR). It must be declared and reported per bank | MLS has 11 voices (narrow), CV / VP are broad. Accent spread is small (HYP) |
| **dev harness (200 / set)** | MLS test (6 speakers), FLEURS `nl_nl` test (364 rows), CV17 test (11,266), VP test (1,137): **4 x 200 = 800** (OBS row counts) | MLS test (10 speakers), FLEURS `pt_br` test (919), CV17 test (9,467); no VP, so a 4th set must be substituted: 600 + X | MLS test (**4 speakers**), FLEURS (758), CV (9,230), VP (1,831) |
| **sealed FINAL (speaker-disjoint)** | MLS dev (6 other speakers), CV17 validation (client_ids disjoint), VP validation (1,230 rows; drop the speakers of our pools), FLEURS `nl_nl` dev (**only 171 rows**), Speech-MASSIVE nl-NL validation (2,033 rows) (OBS [HF-DS]). No mTEDx-nl (OBS: no `nl` in deepdml/mtedx) | the richest: MLS dev, CV val, FLEURS dev (386), **mTEDx-pt train talks** (88,798 rows; talks absent from test / valid and from every Granary id, as IT / FR), Speech-MASSIVE **pt-PT** validation (2,033) | MLS dev (4 speakers), CV val, VP val, FLEURS dev (338), Speech-MASSIVE pl-PL validation (2,033) |
| contamination / leakage | Nemotron was **trained on Granary, MLS, CV, FLEURS and VP** (OBS [NEMO]), so teacher agreement on Granary / VP is not independent of the teacher's own training data. This is recipe-constant (IT / FR had it too). Which splits the teacher saw is UNK, but eval banks are chosen model-free and the teacher never scores an eval clip. The EN parent: no nl data (HYP). VP splits share speakers (OBS, prepare_final_it.py docstring), so drop them by id | same, plus mTEDx-pt vs Granary-pt video overlap, which the existing Granary-id exclusion handles | same |
| expected difficulty | medium. Long compounds fragment under the EN SPE. MLS nl is 1,554 h of only 40 voices, and the test is 6 voices. HYP: harness greedy in the FR range (~20-26), worse than ES | low-medium: Romance, rich data, strong teacher. HYP: close to ES (~17-22) | high: inflection, de-accent merges, a weak teacher, thin MLS voices. HYP: 25-35 |
| **value as a prospective test** | **strong**: a new family, same script, every contract clause exercised without any edit beyond language plumbing. A pass supports "Latin-script EU languages" | weak-to-medium: so close to ES / IT that a pass adds little, and the VP gap forces a contract amendment, which is no longer "frozen" | confounded: a failure would not tell "the recipe" apart from "the language" or "the new pre-map" |
| licence red flags | none in the train mix: MLS CC-BY-4.0, CV CC0, VP CC0, FLEURS CC-BY-4.0, yodas-granary CC-BY-3.0 (loader header), Granary CC-BY-4.0. Speech-MASSIVE eval-only: CC-BY-NC-SA (HYP, check the card). Avoid CGN (restricted TST licence, HYP) | mTEDx is CC-BY-NC-ND (HYP from use in IT / FR): eval-only, as in FR. CORAA / MuPe: NC / ND, so avoid | as nl |

## 4. Recommendation: **Dutch (nl)** for the Factory v1 prospective frozen-recipe validation

Why Dutch and not Portuguese:
- A prospective validation must run the frozen contract **as written**. Portuguese cannot: it has no VP-pt
  transcription, so it would need an amendment such as a VP-weight redistribution or a Granary-for-VP swap.
- Portuguese also adds a BR / PT confound, and its expected outcome is close to the four languages we developed on.
- Dutch needs only the language plumbing that ES needed: a `PRE["nl"]` key, `LANGS["nl"] = ("dutch", 48, "nl_nl")`,
  `FLEURS_CFG["nl"] = "nl_nl"`, and the teacher locale `nl-NL`.
- If the user wants a Romance language regardless, take Portuguese with a pre-registered amendment: §4.4.

### 4.1 Source mix (factory-v1 = A2-type 4 human sources + a G1E-like web source at ~1/3 of the draws)

| source | loader | cap / filter | planned physical h | weight |
|---|---|---|---|---|
| MLS-nl train | prepare_it.py `--lang nl` | 40 h cut over all 40 speakers, sha1 order | 40.0 | A2 0.35 x 0.6733 |
| CV17-nl train | prepare_mix.py | <= 20 clips / speaker, 40 h cap | ~35-40 (HYP) | 0.30 x 0.6733 |
| VP-nl train | prepare_mix.py + a2_vpfilter.sh `LANG_=nl` | <= 60 / speaker, 40 h cap, Nemotron t50 | ~35-40 (HYP) | 0.20 x 0.6733 |
| FLEURS nl_nl train | prepare_mix.py | 1-20 s, `text_ok` | ~7-9 (HYP) | 0.15 x 0.6733 |
| Granary-YODAS nl (`nl000`, `nl100`, spread shards) | prepare_mix.py `--yg-spread` + granary_build.sh `LANG_=nl` | two-teacher agree <= 20 (the FR value), label = teacher text, clips 1-30 s | **~89** (target = FR-G1's 89 h) | 0.3267 |
| **total** | | | **~206-218 h (target 216 ± 10)** | |

Matching the FR G1 physical total (216.3 h) on purpose means the horizon is the FR-H2 one (23,762 steps, about
4.6 passes). It also makes the open "passes vs steps" question moot for this run. That is HYP: it holds only if the
mean clip duration is close to FR's, so compute both readings when the mix is built and freeze the steps.

### 4.2 Eval banks (all built and sha-sealed before any nl score; model-free selection as in prepare_final_it.py)

| bank | composition | clips |
|---|---|---|
| **harness-nl (DEV)** | MLS-nl test 200 (6 speakers) / FLEURS nl_nl test 200 / CV17-nl test 200 / VP-nl test 200 | 800 |
| OOD-nl dev (secondary) | Speech-MASSIVE nl-NL `train_115` (115) + 200 Granary-nl clips from held-out videos with agreement <= 5 (teacher-labelled; flagged as not gold) | ~315 |
| DEV-EOU-nl | the DEV harness clips under the serving protocol, plus an internal-pause class bank as in FR | as FR |
| **FINAL-NL (sealed, one shot)** | MLS-nl dev 300 (6 speakers disjoint from train / test, OBS [MLS]) / CV17-nl validation 300 (client_ids absent from train / test) / VP-nl validation 300 (speakers of our pools and the harness dropped) / FLEURS nl_nl dev 171 (all; sentence-disjoint, no speaker id) / Speech-MASSIVE nl-NL validation 300 (speakers of `train_115` dropped) | ~1,370 |

Text dedup against every training manifest and DEV bank follows the IT / FR rules (1-20 s, no digits, >= 2 words).

### 4.3 Cost estimate (HYP unless marked)

| item | size | notes |
|---|---|---|
| MLS-nl train | 23.8 GB (OBS) | the 40 h cut over all speakers touches all 48 shards, as FR's did (HYP); dev / test 0.4 GB |
| CV17-nl train / val / test | 1.8 GB (OBS) | |
| VP-nl train / val / test | 9.8 GB (OBS) | |
| FLEURS nl_nl | ~2.1 GB (OBS parquet sizes; the tarballs are similar, HYP) | |
| Granary-nl | ~10-20 GB (HYP: ~150 h scored, at 0.06-0.1 GB/h from 155 GB / 866k samples) | |
| Speech-MASSIVE nl val + train_115 | <1 GB (HYP) | |
| **total download** | **~50-60 GB** | parquet deleted after extraction (loader) |
| wall time | ~15-45 min of network at 20-60 MB/s; **~1-3 h CPU** decode / resample / cut (HYP; the FR timeout was 4 h for MLS) | CPU / network only |
| teacher passes (GPU, under the lock) | ~30-60 min Nemotron over ~45 h VP + ~150 h Granary (HYP) | not CPU; the only GPU step before training |
| training | S5, 23,762 steps on GB10: ~4.9 h (HYP: 2 x the S5-FR 11,881-step 8,828 s), plus stage 2 v4 ~15 min | |

### 4.4 Fallback if a Romance language is required: Portuguese with a pre-registered amendment

- **Mix:** MLS-pt 40 h (42 speakers), CV17-pt train <= 20 per speaker (~22-28 h), FLEURS pt_br (~8 h), and
  Granary-pt agree <= 20 (~89 h).
- **The amendment:** VP's 0.20 weight is redistributed pro rata across the human sources. It is declared and frozen
  as "factory-v1a", and it is NOT the frozen v1.
- **FINAL-PT:** MLS dev / CV val / FLEURS dev / mTEDx-pt / Speech-MASSIVE pt-PT.
- **Reporting:** BR and PT are reported separately.
- **Downloads:** ~25-35 GB, smaller than Dutch (HYP).

## 5. What must be frozen before launch (one commit + sealed hashes, before any nl score exists)

1. **The recipe:** the factory-v1 args from `factory-v1-candidate.md`, including:
   - stock init and stock SPE;
   - encoder 4e-4, decoder / joint 3e-4, AdamW, wd 1e-3, bs 16;
   - WSD (steps / 20 warm-up, last 20 % decay), max-dur 30;
   - **S5 sampler**;
   - **steps = 23,762** (or the value the frozen rule gives for the measured physical hours);
   - seed 0, plus seed 1 if the budget allows;
   - stage 2 v4 unchanged.
2. **The text rule:** `textnorm.PRE["nl"] = _ES`, plus unit tests on nl strings (apostrophes, ĳ, ë) and the
   0 % `<unk>` / 100 % round-trip check on the pools.
3. **The data contract:**
   - the source list, caps, per-speaker limits, weights (0.6733 / 0.3267) and seeds;
   - the Granary shard list (`--yg-spread` N) and the agreement threshold 20;
   - VP t50, the teacher locale `nl-NL`, and the 300-clip CLI teacher gate;
   - `data_stats.txt` hours to be within ±10 % of plan, or the pre-declared fallback (more Granary shards) applies.
4. **The banks:** harness-nl, OOD-nl, DEV-EOU-nl and FINAL-NL, built and sealed (sha) **before the training
   starts**, with the cross-bank speaker / text / video disjointness asserted.
5. **The gates, pre-registered numerically:**
   - the plain parent's harness-nl greedy band (proposal, HYP: PASS <= 24, which is FR-H2-like; FAIL > 28);
   - FINAL-NL is a single shot;
   - the EOU bands from IT / FR FINAL (proposal: FE <= 10 %, R >= 95 %, p50 <= 1.0 s);
   - what counts as "factory transfers" vs "language-specific failure". The bands are the user's decision; the
     numbers above are only anchors.
6. **No mid-run tuning:** no change of LR, steps, mix or threshold after the first nl score. Any deviation turns
   the run into development and voids the prospective claim.

## 6. Open UNKNOWNs

- The exact CV / VP / FLEURS hours under the caps: they need a metadata-only `--plan-only` pass on a box (CPU, minutes).
- The Granary-nl agree <= 20 keep rate, and so the shards needed for ~89 h.
- The nl-BE share in each source.
- Which splits Nemotron trained on.
- The Speech-MASSIVE licence text and nl speaker counts.
- Whether the mean clip duration makes 23,762 steps equal to ~4.6 passes.
