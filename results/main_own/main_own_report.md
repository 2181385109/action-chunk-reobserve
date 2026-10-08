# Self-trained ACT analysis report (T10b, implements PREREGISTRATION_OWN.md)

- Registered checkpoint: `/root/train_out/act_own/checkpoints/100000/pretrained_model` (model.safetensors SHA-256 `28631d3abc89d054…`).
- Checkpoint check: all 10 result directories record the registered checkpoint (path and SHA-256 in `config.json`).
- Cells: 9/9 complete (expected 200 episodes per cell, seeds 200000-200199, identical seed set in every cell).
- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI (10000 resamples, RNG seed 20261007); Holm correction over H1' and H3' only (alpha = 0.05); separate from the ACT family H1–H3 and the DP family H4–H5.

## Base-level test (descriptive only, no test)

Self-trained model, k = 100, no perturbation, seeds 1000-1499:

| model | successes | success rate | Wilson 95% CI |
|---|---|---|---|
| self-trained (step 100000) | 416/500 | 0.832 | [0.797, 0.862] |
| pretrained hub model (our reproduction) | 413/500 | 0.826 | [0.790, 0.857] |
| official report | — | 0.830 | — |

Weak-model rule (below 0.40) not triggered.

## Primary hypotheses (confirmatory)

| H | (ACT analogue) | condition | A vs B | n | rate A | rate B | diff (A−B) | 95% CI (paired bootstrap) | only A / only B | p (McNemar) | p (Holm) | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1' | H1 | push0.06 | k=20 vs k=100 | 200 | 0.450 | 0.210 | +0.240 | [+0.160, +0.320] | 62 / 14 | 2.324e-08 | 0.0000 | supported |
| H3' | H3 | delay10 | k=100 vs k=50 | 200 | 0.570 | 0.250 | +0.320 | [+0.220, +0.420] | 95 / 31 | 9.61e-09 | 0.0000 | supported |

Hypothesis H: success rate of A is higher than that of B (two-sided test). "only A / only B" = seeds where only A / only B succeeded (the discordant pairs McNemar uses). "OPPOSITE RESULT" = difference in the opposite direction and Holm-adjusted p < 0.05; reported as such. A floor effect (almost no successes in both cells) shows up as a non-significant result and is not evidence against the ACT finding (PREREGISTRATION_OWN.md §5).

## Success rate, all 9 cells (exploratory description; Wilson 95% CI)

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 145/200 = 0.725 [0.659, 0.782] | 170/200 = 0.850 [0.794, 0.893] | 163/200 = 0.815 [0.755, 0.863] |
| push0.06 | 90/200 = 0.450 [0.383, 0.519] | 84/200 = 0.420 [0.354, 0.489] | 42/200 = 0.210 [0.159, 0.272] |
| delay10 | 0/200 = 0.000 [0.000, 0.019] | 50/200 = 0.250 [0.195, 0.314] | 114/200 = 0.570 [0.501, 0.637] |

## Exploratory: gap change, self-trained vs. pretrained model (no test, not evidence)

gap change = [success(k=20, perturbed) − success(k=100, perturbed)] − [success(k=20, none) − success(k=100, none)], averaged over the seeds, with a paired bootstrap 95% CI (10000 resamples, RNG seed 20261007; same resampled seeds in all four cells). Positive = the short k loses less (or gains more) from the perturbation than the long k does. The pretrained model's values are computed from `results/main` on the same seeds.

| model | condition | n | gap (perturbed) | gap (none) | **gap change** | 95% CI |
|---|---|---|---|---|---|---|
| own | push0.06 | 200 | +0.240 | -0.090 | **+0.330** | [+0.225, +0.430] |
| own | delay10 | 200 | -0.570 | -0.090 | **-0.480** | [-0.575, -0.385] |
| pretrained | push0.06 | 200 | +0.325 | -0.155 | **+0.480** | [+0.370, +0.590] |
| pretrained | delay10 | 200 | -0.475 | -0.155 | **-0.320** | [-0.420, -0.215] |

## Exploratory: agreement of pretrained and self-trained model on the same seeds (descriptive only)

Per seed, both models succeed / only the self-trained / only the pretrained / both fail; agreement = fraction of seeds with the same outcome. No test. (Outcomes are not noise-free under D3D12 rendering, so even a model compared with itself would not agree 100%.)

| item | n | both succeed | only self-trained | only pretrained | both fail | agreement | rate self-trained | rate pretrained |
|---|---|---|---|---|---|---|---|---|
| k20__none | 200 | 104 | 41 | 20 | 35 | 0.695 | 0.725 | 0.620 |
| k50__none | 200 | 131 | 39 | 17 | 13 | 0.720 | 0.850 | 0.740 |
| k100__none | 200 | 139 | 24 | 16 | 21 | 0.800 | 0.815 | 0.775 |
| k20__push0.06 | 200 | 74 | 16 | 28 | 82 | 0.780 | 0.450 | 0.510 |
| k50__push0.06 | 200 | 60 | 24 | 21 | 95 | 0.775 | 0.420 | 0.405 |
| k100__push0.06 | 200 | 21 | 21 | 16 | 142 | 0.815 | 0.210 | 0.185 |
| k20__delay10 | 200 | 0 | 0 | 0 | 200 | 1.000 | 0.000 | 0.000 |
| k50__delay10 | 200 | 15 | 35 | 24 | 126 | 0.705 | 0.250 | 0.195 |
| k100__delay10 | 200 | 77 | 37 | 18 | 68 | 0.725 | 0.570 | 0.475 |
| baseline (k=100, seeds 1000–1499) | 500 | 372 | 44 | 41 | 43 | 0.830 | 0.832 | 0.826 |

## Secondary metrics (descriptive only, no tests)

Mean over all episodes of the cell (successes and failures together; episodes where a metric is undefined are left out of that metric's mean) with bootstrap 95% CI (10000 resamples over episodes, RNG seed 20261007). path_length and jerk are per-episode quantities of the executed action sequence; smoothness is the mean L2 norm of consecutive action differences.

### policy calls per episode

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 17.0 [16.7, 17.3] | 6.6 [6.4, 6.7] | 3.4 [3.3, 3.5] |
| push0.06 | 18.2 [17.8, 18.5] | 7.4 [7.3, 7.5] | 3.9 [3.9, 4.0] |
| delay10 | 20.0 [20.0, 20.0] | 7.9 [7.9, 7.9] | 3.8 [3.8, 3.9] |

### jerk

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 0.0196 [0.0189, 0.0203] | 0.0176 [0.0171, 0.0181] | 0.0166 [0.0162, 0.0170] |
| push0.06 | 0.0200 [0.0190, 0.0211] | 0.0201 [0.0194, 0.0208] | 0.0201 [0.0196, 0.0205] |
| delay10 | 0.0332 [0.0325, 0.0340] | 0.0251 [0.0244, 0.0258] | 0.0199 [0.0195, 0.0203] |

### path_length

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 8.21 [7.91, 8.52] | 8.08 [7.87, 8.30] | 7.78 [7.59, 7.98] |
| push0.06 | 8.06 [7.67, 8.48] | 9.18 [8.86, 9.50] | 9.49 [9.29, 9.68] |
| delay10 | 11.54 [11.29, 11.79] | 11.10 [10.88, 11.32] | 9.21 [9.02, 9.40] |

### smoothness

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 0.0253 [0.0243, 0.0261] | 0.0265 [0.0261, 0.0268] | 0.0253 [0.0251, 0.0256] |
| push0.06 | 0.0231 [0.0220, 0.0243] | 0.0256 [0.0250, 0.0263] | 0.0250 [0.0247, 0.0253] |
| delay10 | 0.0289 [0.0283, 0.0296] | 0.0286 [0.0282, 0.0291] | 0.0263 [0.0260, 0.0266] |

## Push episodes: skipped / truncated counts (all included in the analysis)

skipped_lifted = the cube was already lifted at the push step, no push applied; truncated = the push distance was shortened (table edge and/or arm contact); not_reached = the episode ended before the push step.

| cell | n | pushed | skipped_lifted | truncated (table / arm) | not_reached |
|---|---|---|---|---|---|
| k20__push0.06 | 200 | 200 | 0 | 0 (0 / 0) | 0 |
| k50__push0.06 | 200 | 200 | 0 | 0 (0 / 0) | 0 |
| k100__push0.06 | 200 | 200 | 0 | 0 (0 / 0) | 0 |

## Best k per condition (EXPLORATORY; description only, no test)

| condition | best k | success rate | runner-up | gap (pct. points) |
|---|---|---|---|---|
| none | k=50 | 0.850 | k=100 | 3.5 |
| push0.06 | k=20 | 0.450 | k=50 | 3.0 |
| delay10 | k=100 | 0.570 | k=50 | 32.0 |

Figures: `fig_hypotheses_own.png` (confirmatory), `fig_success_own.png` and `fig_interaction_own.png` (exploratory).
