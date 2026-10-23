# Cheap-specialist factory: trainer throughput (2026-10-23, dgx1 GB10)

## Phase 0 (OBSERVED, feat/trainer-throughput 1a8cb47 vs lw-finetune-tooling 2b5d6c7, ES-1 recipe)
- Flags-off parity: old head twice, new head flags off, new head + `--log-batch-stats 1`: loss and pre-clip encoder
  grad norm IDENTICAL to the last digit at steps 25 / 50 / 75 / 100 (the trainer is deterministic on GB10: noise
  floor 0; the batch-stats logging is transparent).
- `--joint-subbatch 4` (S4): NOT bit-identical on the GPU (step 25 loss 287.13 vs 276.49; step 100 135.65 vs 135.28):
  the maths is exact (CPU parity 1e-12) but the bf16 kernels see other shapes and the summation order changes, and the
  training trajectory diverges from there. Peak VRAM 11.9 GB vs 18.1 GB (-34 %).
- `--profile-steps 180`: R 1,015.1 ms / step (fwd+loss 375.4, bwd 566.3, clip+opt 51.5, idle 22.0); **S4 850.7 ms
  / step (-16.2 %)** (fwd+loss 308.2, bwd 469.7, clip+opt 51.3, idle 21.8). Rule set before the profile: < 5 % ->
  CLOSED, 5-10 % -> another day, > 10 % -> a 1,500-step quality screen now.

## CF-E2 -- S4 quality screen, PRE-REGISTERED (before the run)
One 1,500-step run = CF-E1 arm R's exact command + `--joint-subbatch 4` (same git lineage: flags off is
bit-identical to the trainer that ran R), one eval at the last step, harness-es 200 / set. Reference: CF-E1 R
macro 30.82 (1,019 ms / step). Gates: macro <= 30.82 + 0.5 AND wall / step >= 10 % lower -> PASS (S4 becomes a
candidate default for the next full run, which then needs a full-horizon confirmation); macro > 31.32 -> FAIL
(S4 changes quality at this horizon beyond the screen's tolerance). One seed: a PASS is a screen, not a proof.

### Result: CF-E2 (OBSERVED 2026-10-23 11:57-12:19 UTC, dgx1) -- S4 FAIL by its pre-registered gate
S4 1,500 steps: 846 ms / step (-17.0 % vs R 1,019), peak 12.5 GB; macro VAL **31.50 vs R 30.82 (+0.68 > +0.50 -> FAIL)**;
MLS 26.45 (-0.12), FLEURS 30.97 (+0.96), CV 36.61 (+1.40), VP 31.98 (+0.48).
HYPOTHESIS (not a re-classification): S4 is exact maths; it differs from R only through bf16 kernel-shape noise. Its
+0.68 is therefore an estimate of the trajectory spread a tiny perturbation produces at 1,500 steps -- which would
make single-seed 1,500-step screens unable to resolve effects below ~1 WER (CF-E1's B +0.99 included). GB10 training
is deterministic, so R never measured its own variance.

## CF-E3 -- seed noise floor, PRE-REGISTERED (before the run)
R with `--seed 1` (CF-E1 arm R's exact command otherwise), 1,500 steps, one eval at the end. Rule: |R1 - R0| >= 0.6
-> single-seed 1,500-step screens cannot resolve <= 1 WER: CF-E1 B and CF-E2 S4 become UNRESOLVED (multi-seed or
full horizon needed, no recipe claim either way); |R1 - R0| < 0.3 -> S4's FAIL and B's +0.99 stand as real; in
between -> recorded, both stay as classified, with a noise caveat.
