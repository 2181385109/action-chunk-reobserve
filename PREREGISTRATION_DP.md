# Pre-registration (DP): does the ACT finding on execution length generalise to Diffusion Policy on PushT?

Registered on 2026-10-08, **before any final-evaluation episode of this experiment has been run** (final seeds >= 200000 are blocked for Diffusion Policy runs by `src/chunkre/seed_guard.py` until this file is committed). Repository: `2181385109/chunk-reobserve`. This file is separate from `PREREGISTRATION.md` (ACT, committed 2026-10-07, commit `6c5302c`), which is **not modified**. The ACT hypotheses H1–H3 and their Holm family are unchanged and are not affected by this file.

## 1. Purpose

`PREREGISTRATION.md` tested, on ACT / ALOHA transfer-cube, how the execution length *k* trades off against task success under perturbation. This file registers a **replication on a second model and task** (Diffusion Policy, PushT) with two confirmatory hypotheses whose purpose is to check **whether two of the ACT conclusions generalise** (H1 → H4, H3 → H5). It is a controlled measurement study; no new method is claimed. The hypotheses were **not** chosen by searching the pilot for the largest gap between cells (see §7).

## 2. Setup (fixed)

| Item | Value |
|---|---|
| Model | `lerobot/diffusion_pusht` (Diffusion Policy, official reported success 65.4%) |
| Task | PushT (`gym_pusht`), episode length 300 steps, 10 Hz |
| Library | LeRobot 0.3.3 (pinned in `requirements.lock.txt`) |
| Denoising | DDPM, number of inference steps = the official default (100; not overridden) |
| Execution length | `n_action_steps = k` (`mode: fixed`); upper bound for this model is 15 (= horizon 16 − n_obs_steps 2 + 1) |
| Batch size | B = 50 environments per batch (4 batches per cell) |
| Success | the environment's `is_success` flag at the end of the episode (column `success` of `episodes.csv`) |
| Evaluation / perturbation code | as of commit `92027a9` (T8c). Everything added after it (the DP pilot results, this file, the seed guard, run and analysis scripts) does not change evaluation or perturbation logic |

Reproducibility: for a fixed batch composition (B = 50, seeds 200000–200049, 200050–200099, …) the run was checked to give bit-identical actions to the LeRobot 0.3.3 own evaluation (T8c, check K1). The scene (initial state) is fixed by the episode seed; the diffusion sampling noise comes from the global torch RNG, which is seeded once per batch from the first seed of the batch. Two policies on the same seed therefore share the initial scene but not necessarily the sampling noise. PushT reward columns differ by ~1e-16 between reruns of the same configuration (floating point; check D1: `success` was identical in all 50 compared episodes), and no test uses them.

## 3. Design: 3 values of k × 4 conditions = 12 cells

**Execution length k (3):** k ∈ {4, 8, 15} (`mode: fixed`). k = 8 is the official `n_action_steps`; k = 15 is the maximum the model allows; k = 4 is a clearly shorter value. k = 2 (pilot: floor under delay, ≈ 7 min per 50 episodes) is not in the grid.

**Conditions (4):**

| Condition | Cell tag | Definition |
|---|---|---|
| no perturbation | `none` | — |
| push T block, d = 60 | `push60` | at a uniformly random step in the window [20, 80], in a uniformly random direction, the T block is displaced by d = 60 PushT units (canvas 0–512; the T is 120 × 120 units), block velocity set to zero, no rotation (`PushTee`). The displacement is first shortened if the block would leave the walls (recorded as truncated). If the displaced block would overlap the agent (radius 15), the direction is **redrawn** from the same per-episode RNG, at most 20 times; only if all 20 overlap is the push skipped (`skipped_overlap`). If the episode already succeeded before the push step the push is `not_reached`. |
| action delay 2 steps | `delay2` | the action executed at step t is the policy output of step t−2 (the first steps hold the initial agent position); 10 Hz, i.e. 200 ms |
| action delay 4 steps | `delay4` | same with 4 steps, 400 ms |

Cell names: `<k>__<condition>`, e.g. `k4__push60`, `k15__delay4`, `k8__none`. 3 × 4 = 12 cells.

**Observation noise is not in the confirmatory grid.** In the pilot (§7), image noise σ ∈ {4, 8, 16} (0–255) did not lower DP's success rate at any k (σ = 16 vs no noise: k=4 17/50 vs 12/50, k=8 34/50 vs 26/50, k=15 31/50 vs 32/50; all differences within the pilot's sampling noise and, if anything, upward). With no measurable perturbation effect there is nothing for the execution length to trade off against, and the cells would cost run time (3 more cells × 200 episodes) without answering a question. Stronger noise (σ > 16) was not piloted for DP; if it is wanted it needs its own registration before data are collected.

## 4. Seeds

Final evaluation seeds **200000–200199**: 200 episodes per cell, **the same 200 seeds in every one of the 12 cells** (outcomes of two cells on the same seed are paired). These are the same seed numbers as in the ACT experiment (different task, so no scenes are shared between the two experiments). Seeds 0–9999 were used for debugging and the DP pilot (5000–5049); 10000–199999 are reserved and unused.

## 5. Primary hypotheses (confirmatory)

Primary outcome: **episode success** (binary). Two hypotheses, each a **two-sided** test at α = 0.05, **corrected together with Holm** (family of exactly 2; separate from, and not combined with, the ACT family H1–H3):

- **H4** (does ACT H1 generalise?). Under push d = 60, the success rate of k = 4 is higher than that of k = 15.
- **H5** (does ACT H3 generalise?). Under action delay of 4 steps, the success rate of k = 15 is higher than that of k = 8.

**Test, effect size, Holm, "opposite result": identical to `PREREGISTRATION.md` §5.** Outcomes of the two cells on the 200 shared seeds are paired; b = seeds where only A (first-named) succeeds, c = seeds where only B succeeds; exact McNemar test (b ~ Binomial(b + c, 1/2) under H0; two-sided p = min(1, 2·P(X ≤ min(b, c))); p = 1 if b + c = 0). Effect size = success-rate difference (A − B) with a paired bootstrap 95% CI (resample the 200 seeds with replacement 10,000 times, the same resampled seeds for A and B, 2.5th/97.5th percentiles, fixed RNG seed 20261007). Holm: sort the 2 raw p-values, multiply the i-th smallest by (2 − i + 1), take the running maximum, cap at 1. A hypothesis is **supported** if its Holm-adjusted p < 0.05 and the difference is in the hypothesised direction; if Holm-adjusted p < 0.05 in the **opposite** direction it is recorded as an **"opposite result"** and reported as such; otherwise it is reported as not supported (not significant) and no further inference is drawn.

## 6. Secondary outcomes and exploratory analyses

**Secondary outcomes (descriptive only, no tests):** as in `PREREGISTRATION.md` §6, for every cell the mean over all episodes of (i) policy calls per episode, (ii) jerk, (iii) path_length, (iv) smoothness (definitions in `src/chunkre/metrics.py`), each with a bootstrap 95% CI (10,000 resamples over episodes, RNG seed 20261007; undefined episodes left out, valid number reported). **Additionally:** (v) the **mean maximum coverage** (`max_reward`: the episode's maximum overlap between T block and goal, 0–1), same bootstrap CI. For push cells the numbers of pushed / `skipped_overlap` / `not_reached` / wall-truncated episodes and the redraw counts are reported.

**Exploratory analyses (explicitly labelled exploratory; no corrections, no significance tests, not evidence):**

1. Success-rate table and curves for all 12 cells (Wilson 95% intervals); which k has the highest success rate in each condition (ties listed).
2. **Gap change** (a difference of differences), computed for **both models**. For a short execution length S, a long one L and a perturbed condition C, per seed *i* (outcomes 0/1, paired by seed):
   `gap_change_i = [ s_S(i, C) − s_L(i, C) ] − [ s_S(i, none) − s_L(i, none) ]`.
   The reported number is the mean over the seeds, with a paired bootstrap 95% CI (resample seeds, 10,000 times, RNG seed 20261007; the same resampled seeds in all four cells that enter). It equals (rate_S,C − rate_L,C) − (rate_S,none − rate_L,none). A positive value means the short k loses less (or gains more) from the perturbation than the long k does; a negative value, the opposite. **No significance test is done.**
   - **ACT** (data `results/main/`, seeds 200000–200199): S = k 20, L = k 100; C ∈ {push 0.06 m, noise σ = 16, delay 10}.
   - **DP** (data `results/main_dp/`): S = k 4, L = k 15; C ∈ {push 60, delay 4}.
   - **Status:** the DP gap change is specified here before the DP data exist but after the DP pilot. The **ACT gap change was added after the ACT data had been seen** (post hoc), so it is exploratory in the strong sense and carries no confirmatory weight. Its purpose is to separate "a short k is worse at baseline" from "a short k is more (or less) affected by the perturbation", which the raw H1/H4 comparison does not.

## 7. Basis in the pilot

The design comes from a **pilot** (T8b/T8c; `results/pilot_dp/`, 40 cells, k ∈ {2, 4, 8, 15} × 10 conditions, tuning seeds 5000–5049, 50 episodes per cell; table `results/pilot_dp/pilot_table.md`). Selected numbers (successes out of 50):

| condition | k=4 | k=8 | k=15 |
|---|---|---|---|
| none | 12 | 26 | 32 |
| push 60 | 14 | 28 | 30 |
| noise σ=16 | 17 | 34 | 31 |
| delay 2 | 3 | 15 | 23 |
| delay 4 | 8 | 15 | 18 |

What the pilot shows: (a) DP's success rate depends strongly on k **already without perturbation** (k=4 is far below k=8 and k=15); (b) pushing the T block (20/40/60) and image noise (σ 4/8/16) change success little, within the pilot's sampling noise (Wilson intervals about ±13 points); (c) action delay lowers success clearly, most for short k (k=2 reaches 0/50 at delay 2 and 4, floor); (d) with delay, the longer k tends to be better (delay 4: k=15 18/50 vs k=8 15/50).

**Consequence for the hypotheses, stated in advance.** H4 is the DP analogue of ACT H1 and is registered to test whether that finding generalises. The pilot points the other way: under push 60, k=4 (14/50) is *below* k=15 (30/50), and this gap is as large without perturbation (12/50 vs 32/50). **H4 is therefore expected not to hold, and an "opposite result" (k=15 significantly better than k=4 under push 60) is a realistic outcome.** If it occurs, it will mainly reflect k=4's low baseline, not a push-specific effect; the exploratory gap change (§6) is the analysis that addresses this. H5 is the analogue of ACT H3 (the strongest delay level, long k vs. the next shorter k); the pilot direction agrees (k=15 > k=8 at delay 4) but the pilot gap (3/50) is small and the confirmatory test may well be non-significant. The two hypotheses and their levels follow the structure of ACT H1/H3 (strongest level of the perturbation, extreme k vs. the comparison k); they were not picked as the largest gaps in the pilot (the larger pilot gap k=15 vs k=8 is at delay 2, 23/50 vs 15/50, and is not the registered one). These are pilot numbers on 50 episodes and are **not** evidence; the final evaluation uses entirely new seeds.

## 8. Data rules

Identical to `PREREGISTRATION.md` §8:

- **All episodes count; none is excluded.** Push episodes that were skipped, wall-truncated or `not_reached` are included as they are, and their numbers are reported per cell.
- An interrupted run is **resumed**; batches already completed are not re-run.
- The analysis is run once all 12 cells are complete; runs of the analysis scripts on incomplete data, or on the pilot data (dry runs), are only completeness / code checks and no design decision is based on them.
- Any deviation from this registration is written in the **"Deviations" ("偏离") section of the README**.

## 9. Further registrations (not part of this file)

An adaptive re-observation rule and a self-trained ACT replication will be registered in separate files before the corresponding data are collected. Nothing in this file depends on them.

---

## 中文摘要

**目的：** 检验 ACT 上的两条结论能否推广到另一个模型和任务（Diffusion Policy，PushT）。只测量，不声称新方法。与 `PREREGISTRATION.md`（ACT）完全独立，该文件不改动，ACT 的 H1–H3 及其 Holm 校正族不受影响。

**设置：** 模型 `lerobot/diffusion_pusht`，LeRobot 0.3.3，DDPM 去噪步数为官方默认（100），B=50，回合 300 步。**网格 3×4 = 12 格**：执行长度 k ∈ {4, 8, 15}（fixed）× 条件 {无扰动；推 T 形块 d=60（窗口 [20,80]，与智能体重叠时重抽方向，最多 20 次）；动作延迟 2 步；动作延迟 4 步}。**噪声不进正式网格：** 试跑里噪声 σ=4/8/16 对 DP 的成功率没有可见的降低（σ=16 时 k=4 17/50 对 12/50，k=8 34/50 对 26/50，k=15 31/50 对 32/50），没有扰动效应就没有可权衡的东西，只会多花运行时间；更强的噪声没试过，若要做需另外登记。

**种子：** 200000–200199，每格 200 回合，12 格用同一批种子（按种子配对）；调参/试跑种子是 5000–5049，不重复使用。

**两个主要假设**（成功率，双侧，α=0.05，两者一起做 Holm 校正，与 ACT 的 H1–H3 分开）：**H4**（检验 ACT H1 能否推广）推 d=60 时 k=4 的成功率高于 k=15；**H5**（检验 ACT H3 能否推广）延迟 4 步时 k=15 的成功率高于 k=8。检验方法（按种子配对的 McNemar 精确检验）、bootstrap（10000 次，随机种子 20261007）、“相反结果”的定义与 `PREREGISTRATION.md` 完全相同。

**依据与事先说明：** 试跑显示 DP 的成功率在**无扰动时就强烈依赖 k**（k=4 12/50，k=8 26/50，k=15 32/50）；推块和噪声对成功率几乎没有影响；延迟明显降低成功率，短 k 受害最重。所以 **H4 按试跑方向预计不成立，出现“相反结果”（推 60 时 k=15 显著高于 k=4）是现实的可能**，那主要反映 k=4 的基线就低，而不是推的特殊影响。H5 试跑方向一致（延迟 4 时 k=15 18/50 对 k=8 15/50）但差距小，正式检验也可能不显著。两个假设是为了检验 ACT 结论的推广性，沿用 ACT H1/H3 的结构（最强档扰动、极端 k 对比较 k），不是从试跑里挑最大的差距（试跑里更大的差距在延迟 2 步，23/50 对 15/50，并未登记）。

**次要指标**与 `PREREGISTRATION.md` 相同（策略调用次数、jerk、path_length、smoothness，均值 + bootstrap 区间，不检验），另报告**平均最大覆盖率**。**探索性分析**（明确标注，不做显著性检验）：12 格成功率表/曲线；两个模型各算“差距变化”——每个种子的（短 k − 长 k 在扰动下的成败差）减去（短 k − 长 k 在无扰动下的成败差），取平均并给配对 bootstrap 95% 区间。ACT 用 k=20 对 k=100，条件为推 0.06 / 噪声 16 / 延迟 10；DP 用 k=4 对 k=15，条件为推 60 / 延迟 4。**ACT 的这个分析是看过 ACT 数据之后才加的（事后分析）**，DP 的在 DP 数据出现之前登记、但在试跑之后。

**数据规则：** 与 `PREREGISTRATION.md` 相同——所有回合都计入、不剔除；推块的跳过/截断回合照常计入并单独报告；中断就续跑；试跑数据上的分析脚本运行只是代码演示；任何偏离写进 README 的“偏离”小节。
