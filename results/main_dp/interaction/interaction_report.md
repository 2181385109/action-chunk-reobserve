# Exploratory analysis: gap change under perturbation (T9; PREREGISTRATION_DP.md §6.2)

**EXPLORATORY — no significance test, not evidence.** For each seed, gap change = [success(short k, perturbed) − success(long k, perturbed)] − [success(short k, no perturbation) − success(long k, no perturbation)], averaged over the seeds, with a paired bootstrap 95% CI (resampling seeds, 10000 resamples, RNG seed 20261007). Positive = the short k loses less (or gains more) from the perturbation than the long k does; negative = the opposite. It is the change in the short-minus-long success gap caused by the perturbation, so a short k that is simply worse at baseline does not by itself produce a nonzero value.

## ACT (aloha transfer-cube): k=20 (short) vs k=100 (long)

> This ACT analysis was added **after the ACT data had been seen** (post hoc); it carries no confirmatory weight.

Data: `results/main`, seeds 200000-200199.

| condition | n | short, perturbed | long, perturbed | gap (perturbed) | short, none | long, none | gap (none) | **gap change** | 95% CI (paired bootstrap) |
|---|---|---|---|---|---|---|---|---|---|
| push0.06 | 200 | 0.510 | 0.185 | +0.325 | 0.620 | 0.775 | -0.155 | **+0.480** | [+0.370, +0.590] |
| noise16 | 200 | 0.410 | 0.655 | -0.245 | 0.620 | 0.775 | -0.155 | **-0.090** | [-0.210, +0.030] |
| delay10 | 200 | 0.000 | 0.475 | -0.475 | 0.620 | 0.775 | -0.155 | **-0.320** | [-0.420, -0.215] |

## Diffusion Policy (PushT): k=4 (short) vs k=15 (long)

Data: `results/main_dp`, seeds 200000-200199.

| condition | n | short, perturbed | long, perturbed | gap (perturbed) | short, none | long, none | gap (none) | **gap change** | 95% CI (paired bootstrap) |
|---|---|---|---|---|---|---|---|---|---|
| push60 | 200 | 0.190 | 0.540 | -0.350 | 0.275 | 0.515 | -0.240 | **-0.110** | [-0.245, +0.020] |
| delay4 | 200 | 0.105 | 0.410 | -0.305 | 0.275 | 0.515 | -0.240 | **-0.065** | [-0.185, +0.055] |

Figure: `fig_interaction.png`. The intervals are descriptive; they are not adjusted for multiplicity and no decision (alpha = 0.05) is drawn from them.
