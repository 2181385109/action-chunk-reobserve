# Pre-registration (ADAPTIVE): does an adaptive re-observation rule help the pretrained ACT on the pushed-cube task?

Registered on 2026-10-09, **before any episode of the adaptive rule has been run on a final seed (>= 200000)**. Seeds used with the adaptive rule so far: 0–19 (T11a correctness checks A1–A3, `results/t11_checks/`) and 6000–6049 (threshold tuning, `results/tune_adaptive/`, committed in `303e3df`); both are tuning-range seeds. Final seeds >= 200000 are blocked for `mode: adaptive` by `src/chunkre/seed_guard.py` until this file is committed (model key `act_adaptive`; added after this commit). Repository: `2181385109/chunk-reobserve`. This file is separate from `PREREGISTRATION.md` (ACT, commit `6c5302c`), `PREREGISTRATION_DP.md` (DP, commit `e963f3f`, wording `300bf15`) and `PREREGISTRATION_OWN.md` (self-trained ACT, commit `aac0af2`); those three are **not modified**, and their hypothesis families (H1–H3, H4–H5, H1'/H3') and Holm corrections are not affected by this file.

## 1. Purpose

The main ACT experiment (`results/main/`) found that no fixed execution length *k* is best everywhere: under a pushed cube a short k is better (H1), under action delay a long k is better (H3). This file registers a test of one simple idea: **execute the long plan (k = 100) by default and re-plan early only when the policy itself signals that the plan has become stale**. This is a controlled measurement of a rule of a type that is common ("replan on demand"); no new method is claimed. The rule is tested on the one condition where short k clearly beats long k (push 0.06 m) and, as a cost check, on the no-perturbation condition. Action delay and observation noise are **not** tested for the adaptive rule (out of scope; the rule was tuned on neither; not tested is not evidence either way).

## 2. The rule (fixed; implementation `src/chunkre/adaptive.py`, T11a commit `073134a`; T11b does not change evaluation logic)

Model: `lerobot/act_aloha_sim_transfer_cube_human` only (pretrained hub ACT, chunk size L = 100). LeRobot 0.3.3, GPU rendering (EGL + Mesa D3D12; the evaluation script aborts if the renderer is not D3D12), B = 10, success = `is_success` at the end of the 400-step episode.

- By default the current plan (an action chunk of L = 100 steps) is executed step by step, as with k = 100.
- Every **m** steps since the last re-plan (j = m, 2m, … < 100, where j = number of plan steps already executed), the policy is run once on the current observation, giving a **candidate chunk** (this counts as one policy call) that is **not** executed by default.
- **Difference** = mean over s = 0 … h_cmp−1 of the L2 distance between the candidate's action s and the old plan's action j+s. If fewer than h_cmp steps of the old plan remain, only the overlapping steps are compared (overlap = min(h_cmp, 100 − j)). **The L2 distance is computed on the full 14-dimensional action vector**: 12 joint-target angles (radians) and 2 gripper commands (normalised 0..1); the grippers are not rescaled or removed.
- If **difference > tau** and **at least min_gap steps have passed since the last re-plan (j >= min_gap)**, the plan is replaced by the candidate chunk and the first action of the candidate (index 0) is executed at this step. Otherwise the old plan continues.
- When the plan is used up (j = 100) the policy is run again and the plan replaced as usual (a "block-exhausted" re-plan). The initial plan at step 0 is not counted as a re-plan. A diff-triggered re-plan costs no additional policy call (the candidate had already been computed).
- Because min_gap = 20 > m = 10, the comparison at j = 10 right after a re-plan can never trigger a switch under the rule, but it **still runs the policy and counts as one policy call**.

**Fixed parameters:** **m = 10; tau = 0.2906** (the exact value in `configs/tune_adaptive/ad_m10_q90__none.yaml` and `…__push0.06.yaml`, key `tau: 0.2906`, the 90th percentile of the no-perturbation difference distribution, see §3); h_cmp = 20; min_gap = 20; L = 100. Only the pretrained ACT. These values are written into the run script, which refuses to run if the parameters passed to the evaluation differ from them (§4).

## 3. How the parameters were chosen (tuning on non-final seeds; stated for transparency and for the winner's curse below)

- **Seeds:** 6000–6049 (50 episodes per cell, B = 10, GPU), run by the user and committed in `303e3df` before this file.
- **Grid:** 2 values of m (10, 20) × 5 values of tau × 2 conditions (no perturbation; push 0.06 m, definition as in `PREREGISTRATION.md` §3) = 20 cells, plus 2 reference cells of fixed k = 100 in the same two conditions. The 5 tau values are the 50/75/90/95/99% quantiles of the difference distribution in the no-perturbation condition (T11a check A3, `results/t11_checks/checks.json`, seeds 0–19, m = 10, rule never switching; rounded to 4 decimals): 0.0919, 0.1784, 0.2906, 0.3964, 0.7032. h_cmp = 20, min_gap = 20 were fixed in advance (not tuned).
- **Selection rule (fixed before the tuning data existed; `src/chunkre/tune_adaptive.py::select`, run by `scripts/select_adaptive.py`):** (1) a combination is *eligible* if its no-perturbation success rate is >= (k = 100 no-perturbation success rate − 3 percentage points), compared exactly as fractions; (2) among eligible combinations choose the one with the highest success rate under push 0.06 m; (3) ties: fewer mean policy calls (mean of the two conditions); (4) still tied: earlier in grid order (smaller m, smaller tau); (5) no eligible combination: "no usable combination".
- **Result (`results/tune_adaptive/selection.md`): `ad_m10_q90`** (m = 10, tau = 0.2906). Reference k = 100: no perturbation 40/50 = 80.0%, push 0.06 9/50 = 18.0%, mean policy calls 3.7; eligibility bar 77.0%; 6 of 10 combinations eligible.

| combination | m | tau | no perturbation | push 0.06 | mean policy calls (two conditions) | eligible |
|---|---|---|---|---|---|---|
| ad_m10_q50 | 10 | 0.0919 (q50) | 36/50 = 72.0% | 20/50 = 40.0% | 37.2 | no |
| ad_m10_q75 | 10 | 0.1784 (q75) | 32/50 = 64.0% | 24/50 = 48.0% | 36.5 | no |
| **ad_m10_q90** ← | 10 | 0.2906 (q90) | 41/50 = 82.0% | 20/50 = 40.0% | 35.6 | yes |
| ad_m10_q95 | 10 | 0.3964 (q95) | 39/50 = 78.0% | 14/50 = 28.0% | 35.3 | yes |
| ad_m10_q99 | 10 | 0.7032 (q99) | 43/50 = 86.0% | 13/50 = 26.0% | 34.7 | yes |
| ad_m20_q50 | 20 | 0.0919 (q50) | 32/50 = 64.0% | 25/50 = 50.0% | 18.6 | no |
| ad_m20_q75 | 20 | 0.1784 (q75) | 36/50 = 72.0% | 25/50 = 50.0% | 18.2 | no |
| ad_m20_q90 | 20 | 0.2906 (q90) | 39/50 = 78.0% | 13/50 = 26.0% | 17.7 | yes |
| ad_m20_q95 | 20 | 0.3964 (q95) | 40/50 = 80.0% | 12/50 = 24.0% | 17.4 | yes |
| ad_m20_q99 | 20 | 0.7032 (q99) | 43/50 = 86.0% | 13/50 = 26.0% | 17.4 | yes |

**Winner's curse (stated in advance).** The selected combination's push-0.06 result (20/50 = 40.0%) is the *highest of the 6 eligible combinations*; the other five eligible combinations are at 12–14/50 (24–28%), and the neighbouring settings (m = 10 with tau = q95 or q99, m = 20 with q90) are at 13–14/50. A maximum over several noisy 50-episode estimates is biased upwards, and the margin to its neighbours (a cliff between q90 and q95) is at least partly selection noise. **We therefore expect the adaptive rule's push-0.06 success on the final seeds to be lower than 40%**, and we do not treat the tuning figure as an estimate of the final effect. (The tuning run also shows that settings with lower tau, which are not eligible because they lose > 3 points without perturbation, reach 40–50% under push; the eligibility rule is part of the design, not a post-hoc choice.) The no-perturbation figure (41/50 = 82.0%) is likewise selected (it had to pass the eligibility bar) and is not an estimate of the final value. The final seeds 200000–200199 were not used in the tuning in any form.

**Cost note (stated in advance).** With m = 10 the rule makes about 35.6 policy calls per episode in the tuning cells (mean over the two conditions), against 3.7 for k = 100, about 7 for k = 50 and about 19 for k = 20 in the main experiment. The rule is therefore *not* cheap in calls; this is part of the comparison in §6.

## 4. Main experiment: 2 cells

| Cell | Definition |
|---|---|
| `ad__none` | adaptive rule (m = 10, tau = 0.2906, h_cmp = 20, min_gap = 20), no perturbation |
| `ad__push0.06` | the same rule, cube pushed by 0.06 m; definition **identical to `PREREGISTRATION.md` §3**: at a uniformly random step in the window [20, 100], in a uniformly random horizontal direction, the cube is displaced by 0.06 m (per-episode RNG derived from the seed; `PushCube`) |

**Seeds.** Final evaluation seeds **200000–200199**, 200 episodes per cell, the same 200 seeds as in `results/main/` (so the outcomes of the adaptive cells are paired seed by seed with the main-experiment cells), B = 10, GPU (D3D12) rendering. Outputs in `results/main_adaptive/<cell>/`.

**Controls (not re-run).** `results/main/k100__none` and `results/main/k100__push0.06` (same seeds, same pretrained model, same perturbation code). Check A1 (T11a, `results/t11_checks/`) showed that after the adaptive mode was added the fixed mode still reproduces earlier results bit for bit (CPU rendering; tau = ∞ adaptive = k = 100 on 10/10 episodes, and also identical to the T5 run). Under D3D12 rendering ACT is reproducible only statistically, so the same-seed pairing is exact in the initial scene but not necessarily in every later pixel.

**Run guard.** The run script refuses to run any episode unless (i) this file is committed and (ii) the run parameters (m, tau, h_cmp, min_gap) of every cell exactly equal the values in §2.

## 5. Primary hypotheses (confirmatory)

Primary outcome: **episode success** (binary). Two hypotheses, each on the cells above against the k = 100 control on the same 200 seeds. **No Holm correction** is applied: the claim "the adaptive rule is useful on this task" is made only if **both** HA1 and HA2 hold (an intersection–union test requires no multiplicity correction); each is also reported on its own.

- **HA1 (benefit).** Under push 0.06 m, the success rate of the adaptive rule is higher than that of k = 100.
  - **Test, effect size, "opposite result": identical to `PREREGISTRATION.md` §5.** Outcomes paired by seed; b = seeds where only A (adaptive) succeeds, c = seeds where only B (k = 100) succeeds; exact McNemar test (b ~ Binomial(b + c, 1/2); two-sided p = min(1, 2·P(X ≤ min(b, c))); p = 1 if b + c = 0), **two-sided, α = 0.05**. Effect size = success-rate difference (adaptive − k = 100) with a paired bootstrap 95% CI (resample the 200 seeds with replacement 10,000 times, the same resampled seeds for A and B, 2.5th/97.5th percentiles, RNG seed 20261007). HA1 is **supported** if p < 0.05 and the difference is positive; if p < 0.05 and the difference is negative it is recorded as an **"opposite result"**; otherwise not supported (not significant) and no further inference is drawn.
- **HA2 (no loss without perturbation; non-inferiority).** With no perturbation, the adaptive rule is **not inferior** to k = 100 with a margin of **5 percentage points**: HA2 holds if the **lower limit of the paired bootstrap 95% CI of (adaptive − k = 100) is > −0.05** (same bootstrap as above: 10,000 resamples of the 200 seeds, 2.5th percentile, RNG seed 20261007). The exact McNemar test for this condition is reported **descriptively only** (it is not part of the decision).
- **Conclusion rule.** "The adaptive rule is useful on this task" is claimed only if HA1 is supported **and** HA2 holds. If HA1 is supported but HA2 does not hold, the result is "benefit under push, but non-inferiority without perturbation not shown". If HA1 is not supported, no usefulness is claimed whatever HA2 says.

**Stated in advance (power of HA2).** With 200 paired seeds and about 20% of seeds discordant between two policies, the half-width of the bootstrap 95% CI is about ±6 percentage points, so the lower limit exceeds −5 points only if the observed difference is about +1 point or more. A failure of HA2 therefore means "non-inferiority was not shown at this sample size", not "the adaptive rule is worse". This is a property of the design chosen with the 5-point margin and n = 200 and is accepted.

## 6. Secondary and exploratory analyses (descriptive; no tests, not evidence; no corrections)

1. **Descriptive comparison with the fixed policies** (data in `results/main/`, same seeds): for no perturbation and push 0.06 m, list **k = 20, 50, 100, temporal ensembling (te) and the adaptive rule** with success count, success rate, Wilson 95% interval and mean policy calls per episode. **Stated in advance: k = 50 (no perturbation 74.0%, push 0.06 40.5%, about 7.5 policy calls; `results/main/`) is the cheaper fixed control, and the comparison "adaptive vs k = 50" (success difference with paired bootstrap 95% CI, and calls) is reported whatever the outcome.** The same paired differences are listed for k = 20 and te. No p-values are computed for these comparisons.
2. **Re-plans:** per cell, the mean (and distribution) of the number of **diff-triggered re-plans** per episode (and of block-exhausted re-plans, checks and policy calls), taken from the per-episode columns `n_checks`, `n_replans_diff`, `n_replans_exhaust`, `n_policy_calls` and the logged `adaptive/batch_*.npz`.
3. **Reaction to the push** (cell `ad__push0.06` only): for episodes in which a push was applied (`push_status = pushed`), the distribution (count, median, quartiles, minimum, maximum; fraction within 10 / 20 / 30 steps) of the **number of steps from the push step to the first diff-triggered re-plan at or after the push**; episodes with no diff-triggered re-plan after the push (or that ended first) are counted separately. Block-exhausted re-plans are not counted as reactions.
4. Secondary metrics as in `PREREGISTRATION.md` §6 are reported for the two adaptive cells (policy calls, jerk, path_length, smoothness: mean with bootstrap 95% CI; push skipped / truncated / not-reached counts).

## 7. Basis

Everything the rule's parameters rest on is in §3 (tuning seeds 6000–6049, pretrained ACT, committed before this file) and in the T11a checks (`results/t11_checks/checks.md`: A1 reduction to k = 100, A2 interval = m when tau = 0, A3 difference distributions). The final seeds 200000–200199 have been used for the pretrained model in the main experiment (`results/main/`, known: e.g. no perturbation k = 100 155/200, push 0.06 k = 100 37/200, k = 50 148/200 and 81/200) but **never with the adaptive rule**. No part of the rule, tau or the hypotheses was chosen using the main-experiment outcomes of the adaptive rule (none exist). The selection rule and the eligibility margin (3 points) were fixed before the tuning data existed. The hypotheses HA1/HA2 and the 5-point non-inferiority margin were fixed **after** the tuning results were seen (in this file), but before any adaptive episode on a final seed; the 5-point margin is wider than the 3-point eligibility bar used in the tuning.

## 8. Data rules

Identical to `PREREGISTRATION.md` §8:

- **All episodes count; none is excluded.** Push episodes that were skipped, truncated or ended before the push step are included as they are, and their numbers are reported per cell.
- An interrupted run is **resumed**; batches already completed are not re-run. The run parameters and the code commit are written to each cell's `config.json`.
- The analysis is run once both cells are complete; runs of the analysis script on incomplete data or on tuning data (dry runs) are only code checks and no design decision is based on them.
- Any deviation from this registration is written in the **"Deviations" ("偏离") section of the README**.

---

## 中文摘要

**目的：** 主实验发现“没有哪个固定 k 处处最好”：方块被推时短 k 好（H1），动作延迟时长 k 好（H3）。这里检验一个朴素想法：**默认按 k=100 长计划执行，策略自己发现计划过时时才提前重新规划**。只测量，不声称新方法。只在推 0.06 m（短 k 明显更好的条件）和无扰动（成本检查）下检验；延迟、噪声不检验。与另外三份登记完全独立，那三份不改动，各自的假设族和 Holm 校正不受影响。

**规则（固定）：** 只用预训练 ACT。默认按 k=100 执行；自上次换计划起每 **m=10** 步，用当前画面预测一次候选块（算一次策略调用，默认不执行）；差异 = 候选块前 20 个动作与旧计划接下来 20 个动作的逐步 L2 距离的平均，**按 14 维动作向量算**（12 个关节目标角度，弧度；2 个夹爪指令，0..1，不去掉也不缩放）；旧计划剩余不足 20 步时只比较重叠部分；**差异 > tau=0.2906 且距上次换计划 ≥ 20 步**才换（换成候选块，立即执行它的第 1 个动作）；块用完正常换。h_cmp=20，min_gap=20，L=100。m=10 时换计划后第 10 步的比较按规则不能换，但仍算一次调用。tau 取自试跑配置里 q90 的确切值（`tau: 0.2906`）。

**怎么选出来的：** 试跑种子 6000–6049（每格 50 回合，已于 `303e3df` 提交）：2 个 m（10、20）× 5 个 tau（无扰动差异的 50/75/90/95/99% 分位：0.0919/0.1784/0.2906/0.3964/0.7032，来自 T11a 检查 A3，种子 0–19）× {无扰动，推 0.06} + k=100 参照。选择规则事先写在代码里：无扰动成功率不低于 k=100 的无扰动成功率减 3 个百分点者可用；可用里取推 0.06 成功率最高者；并列取平均调用少者；再并列取网格靠前者。结果 **ad_m10_q90**（完整表格见上）。**赢家诅咒（事先说明）：** 推 0.06 的 20/50 是 6 个可用组合里最高的，其余 5 个只有 12–14/50，所以**预计正式种子上会低于 40%**，试跑数字不当作最终效果的估计。**成本事先说明：** m=10 时每回合约 35.6 次策略调用（k=100 约 3.7，k=50 约 7，k=20 约 19），不省调用。

**正式评估：** 2 格 `ad__none`、`ad__push0.06`（推的定义与 `PREREGISTRATION.md` 相同），种子 200000–200199，每格 200 回合，B=10，GPU D3D12，输出 `results/main_adaptive/`。对照用 `results/main/` 的 `k100__none`、`k100__push0.06`（同一批种子；T11a 的 A1 证明 fixed 模式逐位没变）。运行脚本在登记未提交或参数（m、tau、h_cmp、min_gap）与本文不一致时一个回合都不跑。

**两个主要假设（不做 Holm）：** **HA1** 推 0.06 时自适应成功率高于 k=100：按种子配对的精确 McNemar，双侧 α=0.05；效应量和配对 bootstrap 95% 区间、“相反结果”的定义同 `PREREGISTRATION.md` §5。**HA2** 无扰动时自适应**非劣于** k=100，边际 5 个百分点：配对 bootstrap 95% 区间（自适应 − k100）的下界 > −0.05 即成立；McNemar 只作描述。**两个都成立才说“自适应规则在这个任务上有用”**（交并检验，不需校正）；各自也单独报告。事先说明：200 个配对种子下区间半宽约 ±6 个百分点，HA2 要求观察到的差约 ≥ +1 个百分点才能通过，所以 HA2 不成立只是“在这个样本量下没证明非劣”，不是“更差”。

**描述性对比（探索性，不检验）：** 无扰动和推 0.06 下，k20、k50、k100、te 和自适应的成功率（Wilson 区间）与平均策略调用；**事先写明：k=50（无扰动 74.0%、推 40.5%、约 7.5 次调用）是更便宜的固定对照，无论结果如何都报告“自适应对 k=50”的配对差（bootstrap 区间）和调用次数**。另报告每回合差异触发的换计划次数，以及推 0.06 格里被推之后第一次差异换计划距推的步数分布（没有换计划的回合单独计数）。

**数据规则：** 同 `PREREGISTRATION.md` §8——所有回合都计入、不剔除；中断就续跑；试跑数据上的分析只是代码演示；任何偏离写进 README 的“偏离”小节。
