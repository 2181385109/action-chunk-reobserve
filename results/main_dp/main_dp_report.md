# DP main analysis report (T9, implements PREREGISTRATION_DP.md)

- Cells: 12/12 complete (expected 200 episodes per cell, seeds 200000-200199, identical seed set in every cell).
- Tests: exact two-sided McNemar on seed-paired outcomes; paired bootstrap 95% CI (10000 resamples, RNG seed 20261007); Holm correction over H4–H5 only (alpha = 0.05); separate from the ACT family H1–H3.

## Primary hypotheses (confirmatory)

| H | (ACT analogue) | condition | A vs B | n | rate A | rate B | diff (A−B) | 95% CI (paired bootstrap) | only A / only B | p (McNemar) | p (Holm) | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H4 | H1 | push60 | k=4 vs k=15 | 200 | 0.190 | 0.540 | -0.350 | [-0.430, -0.270] | 13 / 83 | 1.202e-13 | 0.0000 | OPPOSITE RESULT |
| H5 | H3 | delay4 | k=15 vs k=8 | 200 | 0.410 | 0.275 | +0.135 | [+0.055, +0.215] | 50 / 23 | 0.002118 | 0.0021 | supported |

Hypothesis H: success rate of A is higher than that of B (two-sided test). "only A / only B" = seeds where only A / only B succeeded (the discordant pairs McNemar uses). "OPPOSITE RESULT" = difference in the opposite direction and Holm-adjusted p < 0.05; reported as such. The PREREGISTRATION_DP.md §7 states in advance that an opposite result for H4 is realistic (k=4 is far below k=15 already without perturbation); the exploratory gap change (`analyze_interaction.py`) separates the baseline gap from the effect of the perturbation.

## Success rate, all 12 cells (exploratory description; Wilson 95% CI)

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 55/200 = 0.275 [0.218, 0.341] | 112/200 = 0.560 [0.491, 0.627] | 103/200 = 0.515 [0.446, 0.583] |
| push60 | 38/200 = 0.190 [0.142, 0.250] | 121/200 = 0.605 [0.536, 0.670] | 108/200 = 0.540 [0.471, 0.608] |
| delay2 | 32/200 = 0.160 [0.116, 0.217] | 102/200 = 0.510 [0.441, 0.578] | 102/200 = 0.510 [0.441, 0.578] |
| delay4 | 21/200 = 0.105 [0.070, 0.155] | 55/200 = 0.275 [0.218, 0.341] | 82/200 = 0.410 [0.344, 0.479] |

## Secondary metrics (descriptive only, no tests)

Mean over all episodes of the cell (successes and failures together; episodes where a metric is undefined are left out of that metric's mean) with bootstrap 95% CI (10000 resamples over episodes, RNG seed 20261007). path_length and jerk are per-episode quantities of the executed action sequence; smoothness is the mean L2 norm of consecutive action differences; max_reward is the episode's maximum coverage score (0–1).

### policy calls per episode

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 66.1 [63.8, 68.4] | 28.8 [27.5, 30.2] | 16.5 [15.9, 17.1] |
| push60 | 70.5 [69.0, 71.9] | 30.2 [29.1, 31.3] | 16.8 [16.2, 17.3] |
| delay2 | 71.4 [70.0, 72.6] | 31.5 [30.3, 32.6] | 16.6 [16.0, 17.2] |
| delay4 | 72.9 [71.9, 73.8] | 35.1 [34.3, 35.9] | 17.6 [17.1, 18.1] |

### jerk

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 4.2533 [4.1098, 4.4023] | 4.2544 [4.1480, 4.3624] | 4.2278 [4.1462, 4.3096] |
| push60 | 4.3645 [4.2612, 4.4689] | 4.3045 [4.2090, 4.4025] | 4.1312 [4.0468, 4.2140] |
| delay2 | 13.4572 [12.8288, 14.1229] | 7.6125 [7.4165, 7.8100] | 5.8188 [5.6826, 5.9501] |
| delay4 | 27.0977 [24.4721, 29.7432] | 11.8742 [11.5381, 12.2107] | 7.6548 [7.4827, 7.8263] |

### path_length

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 1922.87 [1835.77, 2013.98] | 2064.52 [1952.07, 2175.59] | 2379.29 [2277.76, 2481.72] |
| push60 | 2210.94 [2133.92, 2290.89] | 2250.28 [2155.11, 2344.64] | 2400.90 [2306.92, 2491.76] |
| delay2 | 3028.86 [2882.99, 3177.72] | 2389.31 [2284.47, 2492.95] | 2412.48 [2301.49, 2516.83] |
| delay4 | 5459.19 [5021.99, 5904.55] | 3281.92 [3161.25, 3398.46] | 2777.16 [2670.28, 2881.64] |

### smoothness

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 7.6278 [7.2877, 7.9586] | 9.2367 [8.9850, 9.4769] | 9.7587 [9.5801, 9.9317] |
| push60 | 7.9753 [7.7077, 8.2394] | 9.5112 [9.3100, 9.7090] | 9.6948 [9.5308, 9.8572] |
| delay2 | 10.5914 [10.1456, 11.0530] | 9.7017 [9.4688, 9.9338] | 9.7687 [9.5592, 9.9673] |
| delay4 | 18.6257 [17.1890, 20.0779] | 11.8418 [11.5275, 12.1562] | 10.5991 [10.3819, 10.8143] |

### mean maximum coverage (max_reward)

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 0.924 [0.900, 0.945] | 0.917 [0.891, 0.941] | 0.935 [0.910, 0.957] |
| push60 | 0.927 [0.905, 0.947] | 0.934 [0.909, 0.956] | 0.919 [0.891, 0.943] |
| delay2 | 0.682 [0.637, 0.726] | 0.921 [0.893, 0.946] | 0.914 [0.886, 0.938] |
| delay4 | 0.687 [0.643, 0.731] | 0.780 [0.740, 0.818] | 0.886 [0.855, 0.914] |

## Push episodes: pushed / skipped / truncated / redraw counts (all included in the analysis)

pushed = the push was applied; skipped_overlap = all 20 redrawn directions overlapped the agent, no push applied; not_reached = the episode had already succeeded before the push step; truncated_wall = the displacement was shortened so the block stays inside the walls; redrawn = episodes in which the direction was redrawn at least once.

| cell | n | pushed | skipped_overlap | not_reached | truncated_wall | redrawn (max / mean redraws) |
|---|---|---|---|---|---|---|
| k4__push60 | 200 | 200 | 0 | 0 | 9 | 72 (11 / 0.65) |
| k8__push60 | 200 | 199 | 0 | 1 | 12 | 69 (12 / 0.67) |
| k15__push60 | 200 | 199 | 1 | 0 | 13 | 66 (20 / 0.77) |

## Best k per condition (EXPLORATORY; description only, no test)

| condition | best k | success rate | runner-up | gap (pct. points) |
|---|---|---|---|---|
| none | k=8 | 0.560 | k=15 | 4.5 |
| push60 | k=8 | 0.605 | k=15 | 6.5 |
| delay2 | k=8, k=15 | 0.510 | k=4 | 35.0 |
| delay4 | k=15 | 0.410 | k=8 | 13.5 |

Figures: `fig_hypotheses_dp.png` (confirmatory), `fig_success_dp.png` (exploratory success rates).
