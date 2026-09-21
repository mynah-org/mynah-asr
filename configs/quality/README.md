# `configs/quality/` — what this model scored when it was known good

One JSON per (model, quant, mode, lookahead), written by
`tools/eval/cer_offline.py` and read back by it as a baseline.

## Why a baseline and not a threshold

Every quality gate in this repo used to compare against a constant: 0.20 in
`tools/eval/test_samples.py`, 0.30 in `tools/eval/test_langs.py`, 0.25 in the
streaming harness. A model whose CER goes from 0.13 to 0.19 passes all three
while having got half again as bad. A constant answers "is this usable"; a
baseline answers "did this change break it", and only the second question is a
regression gate.

## Using one

    make cer-baseline MODEL_DIR=models/<pack> QUANT=int8    # write it
    make cer-check    MODEL_DIR=models/<pack> QUANT=int8    # compare against it

`cer-check` exits 1 when the mean CER is more than `--margin` (default 0.02)
above the baseline. The margin is deliberately above the int8-versus-f32
distance this repo has recorded (0.133 -> 0.145 over 34 locales,
`docs/quantization.md`) and below any regression worth shipping.

A comparison is REFUSED, not approximated, when the baseline was taken with a
different model, quant, mode, lookahead or manifest. Two unknowns are not a
comparison.

## What a file does and does not claim

It is the offline CLI path on an idle machine: no server, no concurrency, no
pacing. That is on purpose. A CER measured under load answers two questions at
once — is the model right, and did the server drop audio — and separates
neither. The loaded number belongs to a soak run with `--transcripts`, and this
file is what that number is held against.

Clips in a language the pack does not serve are counted apart from errors: a
refusal is the runtime working, not a defect.

## Provenance

Each file records the model directory name, quant, mode, lookahead and the
manifest path, plus every per-clip row with its hypothesis, so a regression can
be read down to the clip that caused it. The bank itself is identified by path;
when the stress bank exists (`make fetch-stress-bank`) its manifest carries a
`bank_hash`, and a change of bank is a change of era, not a regression.
