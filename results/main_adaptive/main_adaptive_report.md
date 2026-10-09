# Adaptive re-observation rule — analysis report (T11b, implements PREREGISTRATION_ADAPTIVE.md)

- Registered rule: m = 10, tau = 0.2906, h_cmp = 20, min_gap = 20, L = 100; pretrained ACT only.
- Parameter check: all 2 adaptive cells record exactly the registered parameters (config.json).
- Cells: expected 200 episodes per cell, seeds 200000-200199; control = k = 100 cells of `results/main` (same seeds, paired).
- Tests: exact two-sided McNemar on seed-paired outcomes (HA1); paired bootstrap 95% CI (10000 resamples, RNG seed 20261007); non-inferiority margin 0.05 (HA2); alpha = 0.05; **no Holm** (intersection–union).

## Primary hypotheses (confirmatory)

| H | condition | n | adaptive | k=100 | diff (adaptive − k=100) | 95% CI (paired bootstrap) | only adaptive / only k=100 | p (McNemar) | decision |
|---|---|---|---|---|---|---|---|---|---|
| HA1 | push0.06 | 200 | 0.350 | 0.185 | +0.165 | [+0.095, +0.235] | 44 / 11 | 8.699e-06 | supported |
| HA2 | none | 200 | 0.710 | 0.775 | -0.065 | [-0.140, +0.010] | 23 / 36 | 0.1175 (descriptive only) | non-inferiority NOT SHOWN (lower limit ≤ −0.05) |

HA1: success rate of the adaptive rule is higher than that of k = 100 under push 0.06 m (two-sided exact McNemar, alpha = 0.05; "OPPOSITE RESULT" = p < 0.05 with k = 100 better). HA2: with no perturbation the adaptive rule is not inferior to k = 100 by more than 5 percentage points (lower limit of the paired bootstrap 95% CI of adaptive − k=100 is > −0.05); McNemar there is descriptive only. With 200 paired seeds the CI half-width is about ±6 points, so a failure of HA2 means "not shown at this sample size", not "worse" (PREREGISTRATION_ADAPTIVE.md §5).

**Conclusion (registered rule): HA1 supported, but non-inferiority without perturbation (HA2) is not shown.**

## Descriptive comparison with the fixed policies (exploratory; no tests, not evidence)

Success rate (Wilson 95% CI) and mean policy calls per episode. **k = 50 is the cheaper fixed control** (stated in advance in the registration); the difference "adaptive − X" is given with a paired bootstrap 95% CI where the seeds are paired (no p-values).

### none

| policy | success | mean policy calls | adaptive − policy (paired bootstrap 95% CI) |
|---|---|---|---|
| k=20 | 124/200 = 0.620 [0.551, 0.684] | 18.46 | +0.090 [+0.010, +0.170] |
| k=50 **(cheaper fixed control)** | 148/200 = 0.740 [0.675, 0.796] | 6.91 | -0.030 [-0.105, +0.045] |
| k=100 | 155/200 = 0.775 [0.712, 0.827] | 3.39 | -0.065 [-0.140, +0.010] |
| temporal ensemble | 153/200 = 0.765 [0.702, 0.818] | 335.63 | -0.055 [-0.125, +0.015] |
| adaptive | 142/200 = 0.710 [0.644, 0.768] | 34.20 | — |

### push0.06

| policy | success | mean policy calls | adaptive − policy (paired bootstrap 95% CI) |
|---|---|---|---|
| k=20 | 102/200 = 0.510 [0.441, 0.578] | 18.84 | -0.160 [-0.235, -0.080] |
| k=50 **(cheaper fixed control)** | 81/200 = 0.405 [0.339, 0.474] | 7.51 | -0.055 [-0.125, +0.015] |
| k=100 | 37/200 = 0.185 [0.137, 0.245] | 3.91 | +0.165 [+0.095, +0.235] |
| temporal ensemble | 47/200 = 0.235 [0.182, 0.298] | 377.86 | +0.115 [+0.050, +0.185] |
| adaptive | 70/200 = 0.350 [0.287, 0.418] | 37.36 | — |

## Re-plans of the adaptive rule (descriptive)

Per episode: number of diff-triggered re-plans, block-exhausted re-plans, comparisons (checks) and policy calls (`n_policy_calls = 1 + n_replans_exhaust + n_checks`).

| condition | n | mean diff-triggered re-plans | median (max) | episodes with ≥1 | distribution 0 / 1 / 2 / 3 / 4+ | mean block-exhausted | mean checks | mean policy calls |
|---|---|---|---|---|---|---|---|---|
| none | 200 | 4.32 | 4 (14) | 91.0% | 18 / 21 / 38 / 22 / 101 | 1.50 | 31.70 | 34.20 |
| push0.06 | 200 | 6.59 | 7 (15) | 99.5% | 1 / 9 / 19 / 17 / 154 | 1.21 | 35.16 | 37.36 |

## Reaction to the push (descriptive; cell push 0.06 m only)

Episodes with an applied push (`push_status = pushed`): 200. For each, the number of steps from the push step to the first **diff-triggered** re-plan strictly after it (the push is applied at the end of step `push_step`, after the policy chose that step's action, so the policy first sees the pushed scene at `push_step + 1`; block-exhausted re-plans are not reactions).

- Episodes with a diff-triggered re-plan after the push: 199; without (or ended first): 1.
- Episodes in which a diff-triggered re-plan happened exactly at `push_step` (observation before the push, not counted as reaction): 2.
- Steps to the first reaction: min 1, Q1 9.0, median 30.0, Q3 97.5, max 204; within 10 / 20 / 30 steps: 30.2% / 46.7% / 50.3% of the episodes with a reaction.

## Secondary metrics of the adaptive cells (descriptive only, no tests)

Mean over all episodes with bootstrap 95% CI (10000 resamples over episodes, RNG seed 20261007).

| cell | policy calls per episode | jerk | path_length | smoothness |
|---|---|---|---|---|
| ad__none | 34.2 [33.6, 34.9] | 0.0204 [0.0197, 0.0212] | 8.31 [8.06, 8.58] | 0.0247 [0.0241, 0.0253] |
| ad__push0.06 | 37.4 [36.7, 38.0] | 0.0234 [0.0223, 0.0245] | 8.92 [8.53, 9.31] | 0.0242 [0.0232, 0.0251] |

Push episodes in `ad__push0.06` (all included): n = 200, pushed 200, skipped_lifted 0, truncated 0 (table 0 / arm 0), not_reached 0.

Figure: `fig_adaptive.png` (left: HA1/HA2 differences with 95% CI; middle: success vs policy calls; right: steps from push to first diff-triggered re-plan).
