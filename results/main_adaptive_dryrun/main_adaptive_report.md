> **DRY RUN — NOT RESULTS.** run with --dry-run. tuning seeds (< 200000) were used. This report only checks that the analysis code runs. Nothing here is a finding about the adaptive rule.

> Cells taken from other seed sets are NOT paired with the adaptive cells (different seeds): k20 (/mnt/d/xiangmu/chunk-reobserve/results/main), k50 (/mnt/d/xiangmu/chunk-reobserve/results/main), te (/mnt/d/xiangmu/chunk-reobserve/results/main); for them only success rates and policy calls are listed, no paired differences.

# Adaptive re-observation rule — analysis report (T11b, implements PREREGISTRATION_ADAPTIVE.md)

- Registered rule: m = 10, tau = 0.2906, h_cmp = 20, min_gap = 20, L = 100; pretrained ACT only.
- Parameter check: all 2 adaptive cells record exactly the registered parameters (config.json).
- Cells: expected 50 episodes per cell, seeds 6000-6049; control = k = 100 cells of `results/tune_adaptive` (same seeds, paired).
- Tests: exact two-sided McNemar on seed-paired outcomes (HA1); paired bootstrap 95% CI (10000 resamples, RNG seed 20261007); non-inferiority margin 0.05 (HA2); alpha = 0.05; **no Holm** (intersection–union).

## Primary hypotheses (confirmatory)

| H | condition | n | adaptive | k=100 | diff (adaptive − k=100) | 95% CI (paired bootstrap) | only adaptive / only k=100 | p (McNemar) | decision |
|---|---|---|---|---|---|---|---|---|---|
| HA1 | push0.06 | 50 | 0.400 | 0.180 | +0.220 | [+0.060, +0.380] | 15 / 4 | 0.01921 | supported |
| HA2 | none | 50 | 0.820 | 0.800 | +0.020 | [-0.080, +0.120] | 4 / 3 | 1 (descriptive only) | non-inferiority NOT SHOWN (lower limit ≤ −0.05) |

HA1: success rate of the adaptive rule is higher than that of k = 100 under push 0.06 m (two-sided exact McNemar, alpha = 0.05; "OPPOSITE RESULT" = p < 0.05 with k = 100 better). HA2: with no perturbation the adaptive rule is not inferior to k = 100 by more than 5 percentage points (lower limit of the paired bootstrap 95% CI of adaptive − k=100 is > −0.05); McNemar there is descriptive only. With 200 paired seeds the CI half-width is about ±6 points, so a failure of HA2 means "not shown at this sample size", not "worse" (PREREGISTRATION_ADAPTIVE.md §5).

**Conclusion (registered rule): HA1 supported, but non-inferiority without perturbation (HA2) is not shown.**

## Descriptive comparison with the fixed policies (exploratory; no tests, not evidence)

Success rate (Wilson 95% CI) and mean policy calls per episode. **k = 50 is the cheaper fixed control** (stated in advance in the registration); the difference "adaptive − X" is given with a paired bootstrap 95% CI where the seeds are paired (no p-values).

### none

| policy | success | mean policy calls | adaptive − policy (paired bootstrap 95% CI) |
|---|---|---|---|
| k=20 | 124/200 = 0.620 [0.551, 0.684] | 18.46 | not paired (different seeds): rates only |
| k=50 **(cheaper fixed control)** | 148/200 = 0.740 [0.675, 0.796] | 6.91 | not paired (different seeds): rates only |
| k=100 | 40/50 = 0.800 [0.670, 0.888] | 3.38 | +0.020 [-0.080, +0.120] |
| temporal ensemble | 153/200 = 0.765 [0.702, 0.818] | 335.63 | not paired (different seeds): rates only |
| adaptive | 41/50 = 0.820 [0.692, 0.902] | 33.76 | — |

### push0.06

| policy | success | mean policy calls | adaptive − policy (paired bootstrap 95% CI) |
|---|---|---|---|
| k=20 | 102/200 = 0.510 [0.441, 0.578] | 18.84 | not paired (different seeds): rates only |
| k=50 **(cheaper fixed control)** | 81/200 = 0.405 [0.339, 0.474] | 7.51 | not paired (different seeds): rates only |
| k=100 | 9/50 = 0.180 [0.098, 0.308] | 3.94 | +0.220 [+0.060, +0.380] |
| temporal ensemble | 47/200 = 0.235 [0.182, 0.298] | 377.86 | not paired (different seeds): rates only |
| adaptive | 20/50 = 0.400 [0.276, 0.538] | 37.40 | — |

## Re-plans of the adaptive rule (descriptive)

Per episode: number of diff-triggered re-plans, block-exhausted re-plans, comparisons (checks) and policy calls (`n_policy_calls = 1 + n_replans_exhaust + n_checks`).

| condition | n | mean diff-triggered re-plans | median (max) | episodes with ≥1 | distribution 0 / 1 / 2 / 3 / 4+ | mean block-exhausted | mean checks | mean policy calls |
|---|---|---|---|---|---|---|---|---|
| none | 50 | 4.28 | 4 (14) | 90.0% | 5 / 4 / 9 / 5 / 27 | 1.34 | 31.42 | 33.76 |
| push0.06 | 50 | 6.40 | 7 (13) | 100.0% | 0 / 1 / 3 / 6 / 40 | 1.12 | 35.28 | 37.40 |

## Reaction to the push (descriptive; cell push 0.06 m only)

Episodes with an applied push (`push_status = pushed`): 50. For each, the number of steps from the push step to the first **diff-triggered** re-plan strictly after it (the push is applied at the end of step `push_step`, after the policy chose that step's action, so the policy first sees the pushed scene at `push_step + 1`; block-exhausted re-plans are not reactions).

- Episodes with a diff-triggered re-plan after the push: 50; without (or ended first): 0.
- Episodes in which a diff-triggered re-plan happened exactly at `push_step` (observation before the push, not counted as reaction): 1.
- Steps to the first reaction: min 1, Q1 11.5, median 59.5, Q3 88.0, max 205; within 10 / 20 / 30 steps: 24.0% / 42.0% / 42.0% of the episodes with a reaction.

## Secondary metrics of the adaptive cells (descriptive only, no tests)

Mean over all episodes with bootstrap 95% CI (10000 resamples over episodes, RNG seed 20261007).

| cell | policy calls per episode | jerk | path_length | smoothness |
|---|---|---|---|---|
| ad__none | 33.8 [32.5, 35.1] | 0.0212 [0.0192, 0.0241] | 8.40 [7.76, 9.25] | 0.0252 [0.0237, 0.0271] |
| ad__push0.06 | 37.4 [36.3, 38.5] | 0.0223 [0.0206, 0.0241] | 8.61 [7.93, 9.28] | 0.0233 [0.0216, 0.0249] |

Push episodes in `ad__push0.06` (all included): n = 50, pushed 50, skipped_lifted 0, truncated 0 (table 0 / arm 0), not_reached 0.

Figure: `fig_adaptive.png` (left: HA1/HA2 differences with 95% CI; middle: success vs policy calls; right: steps from push to first diff-triggered re-plan).
