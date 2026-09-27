# samples/stress-fr — French stress bank for the streaming load harness

The French counterpart of `samples/stress-en`, built by the same tool with
`tools/fetch_stress_bank.py --config fr_fr --lang fr --out samples/stress-fr`
(seed 42, class bounds 8/20 s): 1587 clips, 529 per class, 360.6 min, 660 MiB of
PCM16, `bank_hash sha256:667bd703f076a2e7...`. As for English, the long class is
mostly concatenations of FLEURS recordings with 0.6 s of silence; `manifest.json`
names every source recording.

**The audio is not in the repository.** `samples/stress-fr/**/*.wav` is gitignored;
`manifest.json`, `manifest-validated.json` and this README are tracked.

## Licence and attribution

Clips come from **[FLEURS](https://huggingface.co/datasets/google/fleurs)** (Google,
[Conneau et al. 2022](https://arxiv.org/abs/2205.12446)), **CC-BY 4.0**, re-encoded
to 16 kHz mono PCM16, content unmodified (see `samples/stress-en/README.md` for the
full provenance statement, which applies here unchanged).

## Validation (`manifest-validated.json`)

`tools/bench/bank_trust.py` checked every clip (2026-09-25, Nemotron 3.5 int8):
1143 kept, 444 excluded, each with its reason in the manifest's `validated`
block: peak below -30 dBFS 306, no language tag emitted under `auto` 129,
reference without French orthography 35, offline `lang=fr` WER above 0.5 19,
detected as English 1. Offline `lang=fr` WER mean 0.0976 over all clips, 0.0928
over the kept ones. The per-clip record (`trust.json`, with transcripts) is
evidence and is not tracked. Protocol and results:
`.work/french-validation-20260925.md`.
