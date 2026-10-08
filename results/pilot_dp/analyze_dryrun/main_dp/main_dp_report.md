> **DRY RUN — NOT RESULTS.** This report was produced from the tuning-seed pilot data (seeds 5000-5049, n = 50 per cell) only to check that the analysis code runs. Nothing here is a finding.

# DP main analysis report (T9, implements PREREGISTRATION_DP.md)

- Cells: 12/12 complete (expected 50 episodes per cell, seeds 5000-5049, identical seed set in every cell).
- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI (10000 resamples, RNG seed 20261007); Holm correction over H4–H5 only (alpha = 0.05); separate from the ACT family H1–H3.

## Primary hypotheses (confirmatory)

| H | (ACT analogue) | condition | A vs B | n | rate A | rate B | diff (A−B) | 95% CI (paired bootstrap) | only A / only B | p (McNemar) | p (Holm) | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H4 | H1 | push60 | k=4 vs k=15 | 50 | 0.280 | 0.600 | -0.320 | [-0.500, -0.120] | 6 / 22 | 0.003719 | 0.0074 | OPPOSITE RESULT |
| H5 | H3 | delay4 | k=15 vs k=8 | 50 | 0.360 | 0.300 | +0.060 | [-0.120, +0.220] | 11 / 8 | 0.6476 | 0.6476 | not significant |

Hypothesis H: success rate of A is higher than that of B (two-sided test). "only A / only B" = seeds where only A / only B succeeded (the discordant pairs McNemar uses). "OPPOSITE RESULT" = difference in the opposite direction and Holm-adjusted p < 0.05; reported as such. The PREREGISTRATION_DP.md §7 states in advance that an opposite result for H4 is realistic (k=4 is far below k=15 already without perturbation); the exploratory gap change (`analyze_interaction.py`) separates the baseline gap from the effect of the perturbation.

## Success rate, all 12 cells (exploratory description; Wilson 95% CI)

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 12/50 = 0.240 [0.143, 0.374] | 26/50 = 0.520 [0.385, 0.652] | 32/50 = 0.640 [0.501, 0.759] |
| push60 | 14/50 = 0.280 [0.175, 0.417] | 28/50 = 0.560 [0.423, 0.688] | 30/50 = 0.600 [0.462, 0.724] |
| delay2 | 3/50 = 0.060 [0.021, 0.162] | 15/50 = 0.300 [0.191, 0.438] | 23/50 = 0.460 [0.330, 0.596] |
| delay4 | 8/50 = 0.160 [0.083, 0.285] | 15/50 = 0.300 [0.191, 0.438] | 18/50 = 0.360 [0.241, 0.499] |

## Secondary metrics (descriptive only, no tests)

Mean over all episodes of the cell (successes and failures together; episodes where a metric is undefined are left out of that metric's mean) with bootstrap 95% CI (10000 resamples over episodes, RNG seed 20261007). path_length and jerk are per-episode quantities of the executed action sequence; smoothness is the mean L2 norm of consecutive action differences; max_reward is the episode's maximum coverage score (0–1).

### policy calls per episode

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 67.6 [63.1, 71.4] | 29.2 [26.3, 31.9] | 15.9 [14.6, 17.2] |
| push60 | 68.4 [64.9, 71.7] | 30.9 [28.7, 33.0] | 17.1 [16.1, 18.1] |
| delay2 | 74.1 [72.9, 75.0] | 34.3 [32.3, 36.0] | 17.2 [16.0, 18.3] |
| delay4 | 73.2 [71.2, 74.7] | 35.5 [34.0, 36.8] | 18.2 [17.2, 19.0] |

### jerk

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 4.2629 [4.0404, 4.4937] | 4.3007 [4.0844, 4.5189] | 4.1796 [4.0152, 4.3496] |
| push60 | 4.2279 [4.0049, 4.4466] | 4.2005 [3.9844, 4.4185] | 4.2089 [4.0502, 4.3655] |
| delay2 | 14.3785 [12.7103, 16.4832] | 7.3626 [7.0645, 7.6632] | 5.8917 [5.6557, 6.1300] |
| delay4 | 25.5898 [20.2937, 31.3590] | 11.5144 [11.0501, 11.9854] | 7.8114 [7.4909, 8.1414] |

### path_length

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 2038.62 [1859.53, 2226.01] | 2191.57 [1956.00, 2425.30] | 2291.71 [2072.32, 2504.55] |
| push60 | 2140.00 [1973.34, 2307.48] | 2280.43 [2088.84, 2471.62] | 2465.10 [2292.68, 2629.73] |
| delay2 | 3294.31 [2970.11, 3687.27] | 2510.74 [2333.96, 2678.93] | 2495.31 [2303.50, 2677.70] |
| delay4 | 5249.34 [4372.21, 6193.82] | 3291.90 [3101.29, 3470.59] | 2906.94 [2709.90, 3094.67] |

### smoothness

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 7.7217 [7.1435, 8.3191] | 9.5901 [9.1421, 10.0249] | 9.7738 [9.4409, 10.1037] |
| push60 | 8.0117 [7.4230, 8.5807] | 9.4475 [8.9832, 9.8840] | 9.7839 [9.4771, 10.0842] |
| delay2 | 11.1052 [10.0446, 12.3909] | 9.3536 [8.9693, 9.7296] | 9.7881 [9.4778, 10.0861] |
| delay4 | 17.8368 [14.9533, 20.9395] | 11.7439 [11.2901, 12.1970] | 10.7443 [10.3648, 11.1210] |

### mean maximum coverage (max_reward)

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 0.913 [0.858, 0.958] | 0.927 [0.882, 0.966] | 0.924 [0.866, 0.970] |
| push60 | 0.923 [0.876, 0.963] | 0.928 [0.879, 0.970] | 0.911 [0.848, 0.962] |
| delay2 | 0.562 [0.462, 0.662] | 0.912 [0.862, 0.956] | 0.897 [0.834, 0.952] |
| delay4 | 0.677 [0.577, 0.772] | 0.783 [0.711, 0.852] | 0.866 [0.800, 0.926] |

## Push episodes: pushed / skipped / truncated / redraw counts (all included in the analysis)

pushed = the push was applied; skipped_overlap = all 20 redrawn directions overlapped the agent, no push applied; not_reached = the episode had already succeeded before the push step; truncated_wall = the displacement was shortened so the block stays inside the walls; redrawn = episodes in which the direction was redrawn at least once.

| cell | n | pushed | skipped_overlap | not_reached | truncated_wall | redrawn (max / mean redraws) |
|---|---|---|---|---|---|---|
| k4__push60 | 50 | 50 | 0 | 0 | 2 | 21 (5 / 0.80) |
| k8__push60 | 50 | 50 | 0 | 0 | 1 | 16 (6 / 0.56) |
| k15__push60 | 50 | 50 | 0 | 0 | 3 | 19 (6 / 0.74) |

## Best k per condition (EXPLORATORY; description only, no test)

| condition | best k | success rate | runner-up | gap (pct. points) |
|---|---|---|---|---|
| none | k=15 | 0.640 | k=8 | 12.0 |
| push60 | k=15 | 0.600 | k=8 | 4.0 |
| delay2 | k=15 | 0.460 | k=8 | 16.0 |
| delay4 | k=15 | 0.360 | k=8 | 6.0 |

Figures: `fig_hypotheses_dp.png` (confirmatory), `fig_success_dp.png` (exploratory success rates).
