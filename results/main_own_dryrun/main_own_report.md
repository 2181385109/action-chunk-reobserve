> **DRY RUN — NOT RESULTS.** run with --dry-run. 10 of 10 result directories do not record the registered self-trained checkpoint (e.g. k20__none: policy.repo = 'lerobot/act_aloha_sim_transfer_cube_human', not the registered checkpoint). This report only checks that the analysis code runs. Nothing here is a finding about the self-trained model.

# Self-trained ACT analysis report (T10b, implements PREREGISTRATION_OWN.md)

- Registered checkpoint: `/root/train_out/act_own/checkpoints/100000/pretrained_model` (model.safetensors SHA-256 `28631d3abc89d054…`).
- **Checkpoint check failed** for 10 result directories (k20__none, k50__none, k100__none, k20__push0.06, k50__push0.06, k100__push0.06, k20__delay10, k50__delay10, k100__delay10, own_baseline): policy.repo = 'lerobot/act_aloha_sim_transfer_cube_human', not the registered checkpoint.
- Cells: 9/9 complete (expected 200 episodes per cell, seeds 200000-200199, identical seed set in every cell).
- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI (10000 resamples, RNG seed 20261007); Holm correction over H1' and H3' only (alpha = 0.05); separate from the ACT family H1–H3 and the DP family H4–H5.

## Base-level test (descriptive only, no test)

Self-trained model, k = 100, no perturbation, seeds 1000-1499:

| model | successes | success rate | Wilson 95% CI |
|---|---|---|---|
| self-trained (step 100000) | 413/500 | 0.826 | [0.790, 0.857] |
| pretrained hub model (our reproduction) | 413/500 | 0.826 | [0.790, 0.857] |
| official report | — | 0.830 | — |

Weak-model rule (below 0.40) not triggered.

## Primary hypotheses (confirmatory)

| H | (ACT analogue) | condition | A vs B | n | rate A | rate B | diff (A−B) | 95% CI (paired bootstrap) | only A / only B | p (McNemar) | p (Holm) | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1' | H1 | push0.06 | k=20 vs k=100 | 200 | 0.510 | 0.185 | +0.325 | [+0.245, +0.405] | 76 / 11 | 4.211e-13 | 0.0000 | supported |
| H3' | H3 | delay10 | k=100 vs k=50 | 200 | 0.475 | 0.195 | +0.280 | [+0.190, +0.365] | 74 / 18 | 3.101e-09 | 0.0000 | supported |

Hypothesis H: success rate of A is higher than that of B (two-sided test). "only A / only B" = seeds where only A / only B succeeded (the discordant pairs McNemar uses). "OPPOSITE RESULT" = difference in the opposite direction and Holm-adjusted p < 0.05; reported as such. A floor effect (almost no successes in both cells) shows up as a non-significant result and is not evidence against the ACT finding (PREREGISTRATION_OWN.md §5).

## Success rate, all 9 cells (exploratory description; Wilson 95% CI)

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 124/200 = 0.620 [0.551, 0.684] | 148/200 = 0.740 [0.675, 0.796] | 155/200 = 0.775 [0.712, 0.827] |
| push0.06 | 102/200 = 0.510 [0.441, 0.578] | 81/200 = 0.405 [0.339, 0.474] | 37/200 = 0.185 [0.137, 0.245] |
| delay10 | 0/200 = 0.000 [0.000, 0.019] | 39/200 = 0.195 [0.146, 0.255] | 95/200 = 0.475 [0.407, 0.544] |

## Exploratory: gap change, self-trained vs. pretrained model (no test, not evidence)

gap change = [success(k=20, perturbed) − success(k=100, perturbed)] − [success(k=20, none) − success(k=100, none)], averaged over the seeds, with a paired bootstrap 95% CI (10000 resamples, RNG seed 20261007; same resampled seeds in all four cells). Positive = the short k loses less (or gains more) from the perturbation than the long k does. The pretrained model's values are computed from `results/main` on the same seeds.

| model | condition | n | gap (perturbed) | gap (none) | **gap change** | 95% CI |
|---|---|---|---|---|---|---|
| own | push0.06 | 200 | +0.325 | -0.155 | **+0.480** | [+0.370, +0.590] |
| own | delay10 | 200 | -0.475 | -0.155 | **-0.320** | [-0.420, -0.215] |
| pretrained | push0.06 | 200 | +0.325 | -0.155 | **+0.480** | [+0.370, +0.590] |
| pretrained | delay10 | 200 | -0.475 | -0.155 | **-0.320** | [-0.420, -0.215] |

## Exploratory: agreement of pretrained and self-trained model on the same seeds (descriptive only)

> In this dry run both inputs are the same pretrained-model data, so the agreement is trivially 100% — it only checks that the code runs.

Per seed, both models succeed / only the self-trained / only the pretrained / both fail; agreement = fraction of seeds with the same outcome. No test. (Outcomes are not noise-free under D3D12 rendering, so even a model compared with itself would not agree 100%.)

| item | n | both succeed | only self-trained | only pretrained | both fail | agreement | rate self-trained | rate pretrained |
|---|---|---|---|---|---|---|---|---|
| k20__none | 200 | 124 | 0 | 0 | 76 | 1.000 | 0.620 | 0.620 |
| k50__none | 200 | 148 | 0 | 0 | 52 | 1.000 | 0.740 | 0.740 |
| k100__none | 200 | 155 | 0 | 0 | 45 | 1.000 | 0.775 | 0.775 |
| k20__push0.06 | 200 | 102 | 0 | 0 | 98 | 1.000 | 0.510 | 0.510 |
| k50__push0.06 | 200 | 81 | 0 | 0 | 119 | 1.000 | 0.405 | 0.405 |
| k100__push0.06 | 200 | 37 | 0 | 0 | 163 | 1.000 | 0.185 | 0.185 |
| k20__delay10 | 200 | 0 | 0 | 0 | 200 | 1.000 | 0.000 | 0.000 |
| k50__delay10 | 200 | 39 | 0 | 0 | 161 | 1.000 | 0.195 | 0.195 |
| k100__delay10 | 200 | 95 | 0 | 0 | 105 | 1.000 | 0.475 | 0.475 |
| baseline (k=100, seeds 1000–1499) | 500 | 413 | 0 | 0 | 87 | 1.000 | 0.826 | 0.826 |

## Secondary metrics (descriptive only, no tests)

Mean over all episodes of the cell (successes and failures together; episodes where a metric is undefined are left out of that metric's mean) with bootstrap 95% CI (10000 resamples over episodes, RNG seed 20261007). path_length and jerk are per-episode quantities of the executed action sequence; smoothness is the mean L2 norm of consecutive action differences.

### policy calls per episode

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 18.5 [18.2, 18.7] | 6.9 [6.8, 7.0] | 3.4 [3.3, 3.5] |
| push0.06 | 18.8 [18.6, 19.1] | 7.5 [7.4, 7.6] | 3.9 [3.9, 4.0] |
| delay10 | 20.0 [20.0, 20.0] | 8.0 [7.9, 8.0] | 3.9 [3.8, 3.9] |

### jerk

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 0.0209 [0.0203, 0.0215] | 0.0209 [0.0202, 0.0216] | 0.0198 [0.0193, 0.0204] |
| push0.06 | 0.0216 [0.0207, 0.0225] | 0.0224 [0.0216, 0.0232] | 0.0241 [0.0235, 0.0247] |
| delay10 | 0.0282 [0.0271, 0.0293] | 0.0276 [0.0269, 0.0283] | 0.0232 [0.0226, 0.0238] |

### path_length

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 8.77 [8.55, 8.99] | 8.57 [8.32, 8.83] | 7.98 [7.74, 8.23] |
| push0.06 | 8.61 [8.28, 8.95] | 9.13 [8.80, 9.46] | 9.82 [9.59, 10.03] |
| delay10 | 8.81 [8.44, 9.20] | 11.05 [10.84, 11.27] | 9.58 [9.36, 9.81] |

### smoothness

| condition | k=20 | k=50 | k=100 |
|---|---|---|---|
| none | 0.0243 [0.0237, 0.0249] | 0.0262 [0.0258, 0.0266] | 0.0255 [0.0252, 0.0258] |
| push0.06 | 0.0234 [0.0225, 0.0243] | 0.0251 [0.0244, 0.0258] | 0.0259 [0.0256, 0.0262] |
| delay10 | 0.0221 [0.0211, 0.0231] | 0.0281 [0.0277, 0.0286] | 0.0264 [0.0261, 0.0268] |

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
| none | k=100 | 0.775 | k=50 | 3.5 |
| push0.06 | k=20 | 0.510 | k=50 | 10.5 |
| delay10 | k=100 | 0.475 | k=50 | 28.0 |

Figures: `fig_hypotheses_own.png` (confirmatory), `fig_success_own.png` and `fig_interaction_own.png` (exploratory).
