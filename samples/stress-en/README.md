# samples/stress-en — English stress bank for the streaming load harness

The bank `tools/bench/stream_load.py` plays in a SOAK. `samples/` holds a handful of
clips chosen for quality tests; a load run needs a *population*: the SOAK path stratifies
its schedule over duration classes (`classify()`: short < 8 s ≤ medium < 20 s ≤ long) and
plays one clip of each class per round, so three clips cannot fill three classes.

**The audio is not in the repository.** `samples/stress-en/**/*.wav` is gitignored;
`manifest.json` and this README are tracked. The bank is regenerated from FLEURS, and the
manifest is what makes a regenerated bank verifiably the same one.

## Licence and attribution

Clips come from **[FLEURS](https://huggingface.co/datasets/google/fleurs)** (Google,
[Conneau et al. 2022](https://arxiv.org/abs/2205.12446)), **CC-BY 4.0** — the same source,
licence and provenance as the committed `samples/`. Content is unmodified: the audio is
re-encoded from the published float32 WAV to 16 kHz mono PCM16 (the only format the runtime
and the harness read), and clips of the `long` class are concatenations of FLEURS
recordings with 0.6 s of silence between them — `manifest.json` names every source
recording, its split and its FLEURS id, so any clip can be traced back. Anything published
from a run on this bank must carry the FLEURS attribution and the CC-BY 4.0 notice.

## Regenerate

    make fetch-stress-bank                                   # ~1.8 GB transfer, ~640 MB on disk
    make fetch-stress-bank STRESS_BANK_ARGS="--dry-run"      # the plan only, no audio
    make fetch-stress-bank STRESS_BANK_ARGS="--per-class 64" # a laptop-sized bank (~80 MB)

`tools/fetch_stress_bank.py` is offline tooling (uv, like the rest of `tools/`); nothing at
runtime reads it. It is deterministic — one seed (default 42, the harness's own) fixes
which recording lands in which clip under which name — and idempotent: a clip whose sha256
matches the manifest is neither rebuilt nor re-downloaded, so a second run touches no audio
and a partial bank resumes. The float32 → PCM16 conversion is done in the tool rather than
by libsndfile, so the bytes do not depend on the writer's version.

Default size: 529 clips per class, ~5.8 h of audio. The number comes from the run the bank
must cover — a 10-minute SOAK at concurrency 32 in which no clip is played twice anywhere
in the fleet; the arithmetic, and the simulation that checks it against the harness's own
`build_bank()` / `stratified_schedule()`, are in the tool's header.

## Layout and manifest

    short/  medium/  long/     16 kHz mono PCM16, one clip per file
    manifest.json              provenance + one entry per clip

Top level: `source`, `url`, `licence`, `attribution`, `fetched_utc`, `tool`,
`tool_version`, `config`, `splits`, `seed`, `per_class`, `class_bounds_s`, `long_min_s`,
`pause_sec`, `soak_profile`, `totals`, and **`bank_hash`** — a sha256 over the sorted
per-clip sha256 values, so a run records *which* bank it used in one field. Per clip:
`file`, `class`, `sha256`, `duration_sec`, `sample_rate`, `text`, `text_norm`, `fleurs_id`
(`fleurs_ids` and `parts` for a composed clip), `split`, `gender`, `peak_dbfs`, `rms_dbfs`.

Use it, with the bounds the bank was built for:

    python3 tools/bench/stream_load.py --mode soak --streams 32 --duration 600 \
        --warmup 30 --window 60 --bank short,medium,long --class-bounds 8,20 --seed 42 \
        --clips samples/stress-en/*/*.wav --port 8090 --json soak-32.json

`text` is the reference for a CER or identity gate; build the harness's `--reference` file
from the manifest (`{clip_path: text}`) rather than by hand, so paths and text cannot drift.

## What this bank is NOT

FLEURS is **read speech**: volunteers reading Wikipedia sentences, one sentence at a time,
in clean recording conditions, `en_us`. That is a deliberate choice — it makes the bank
reproducible, licensed and traceable — but it bounds what may be claimed from a run on it:

- **No conversational or spontaneous speech.** No disfluencies, repairs, hesitations,
  interruptions, crosstalk or overlapping speakers. **A claim about conversational ASR
  cannot be made from this bank**, whatever its CER. That is a known gap, not an oversight:
  closing it needs a differently licensed corpus and a separate board item.
- **No long-form discourse.** The `long` class is *built*: only 68 of the 3,555 en_us
  recordings reach 20 s, so long clips are shorter recordings concatenated with 0.6 s of
  silence. They exercise long streams, cache-aware decoding across many chunks and
  finalisation — they are not a naturally long utterance, and the silences are real
  sentence boundaries a segmenter can exploit.
- **No channel or noise diversity.** No telephony band, no far-field, no music or babble,
  no packet loss. A robustness claim needs its own material.
- **Levels are not normalised.** FLEURS recordings differ by ~35 dB, and Nemotron's front
  end does not normalise per utterance, so quiet clips are genuinely quiet for the model
  (the fetcher reports how many clips peak below -30 dBFS). Filter on `peak_dbfs` for a
  quality gate; for a throughput SOAK it does not matter.
- **One variety of English.** `en_us` only, and FLEURS' speaker pool, so it says nothing
  about accents outside it.

For scheduling, admission, latency under concurrency and drift — what the serving work is
being judged on — the bank is the right instrument. For quality claims it is a
read-speech corpus and must be described as one.
