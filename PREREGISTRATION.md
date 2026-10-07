# Pre-registration: how long should an action chunk be executed before re-observing? (ACT on ALOHA transfer-cube, perturbation grid)

Registered on 2026-10-07, **before any final-evaluation episode has been run** (final seeds >= 200000 are blocked by `src/chunkre/seed_guard.py` until this file is committed). Repository: `2181385109/chunk-reobserve`.

## 1. Research question

Imitation-learning policies such as ACT predict a chunk of actions and execute the first *k* of them before observing and predicting again. **How do the execution length *k* (and temporal ensembling, which re-predicts every step) trade off against task success under controlled perturbations — a pushed object, noisy observations, and delayed actions?** This is a controlled measurement study; it does not claim a new method.

## 2. Setup (fixed)

| Item | Value |
|---|---|
| Model | `lerobot/act_aloha_sim_transfer_cube_human` (ACT, ALOHA sim transfer-cube; official reported success 83%) |
| Library | LeRobot 0.3.3 (pinned in `requirements.lock.txt`) |
| Rendering | GPU (EGL + Mesa D3D12, NVIDIA RTX 4060 Laptop); the evaluation script aborts if the renderer is not D3D12 |
| Batch size | B = 10 environments per batch |
| Success | the environment's `is_success` flag at the end of the episode (column `success` of `episodes.csv`); episode length 400 steps |
| Evaluation / perturbation code | as of commit `4b6aadf` (T6a). T6b adds only analysis and run scripts, a statistics module and grid definitions; it does not change evaluation or perturbation logic |

Note: ACT under D3D12 rendering is reproducible only statistically, not bit-for-bit; a re-run with the same seeds can differ in individual episodes.

## 3. Design: 4 policies × 10 conditions = 40 cells

**Policies (4):** fixed execution length k ∈ {20, 50, 100} (`mode: fixed`), and temporal ensembling `te` (`temporal_ensemble_coeff = 0.01`, re-predicts every step).

**Conditions (10):**

| Perturbation | Levels | Definition |
|---|---|---|
| none | — | no perturbation |
| push cube | 0.02, 0.04, 0.06 m | at a uniformly random step in the window [20, 100], in a uniformly random horizontal direction, the cube is displaced by the given distance (per-episode RNG derived from the seed) |
| observation noise | **σ ∈ {4, 8, 16}** (pixel units, 0–255) | independent Gaussian noise added to every pixel of every camera image at every step, `clip(rint(img + σ·N(0,1)), 0, 255)` |
| action delay | 3, 5, 10 steps | the action executed at step t is the policy output of step t−d (the first d steps hold the initial joint positions); 50 Hz, i.e. 60 / 100 / 200 ms |

The grid has 4 × 10 = 40 cells. Cell names: `<policy>__<condition>`, e.g. `k20__push0.06`, `te__noise16`, `k100__delay10`, `k50__none`.

**Why these levels (from the pilot, see §7):** noise levels were chosen by a rule fixed before the supplementary σ=32 pilot run: *if all four policies (k=20, 50, 100, te) have ≤ 2/20 successes at σ=32, the final noise levels are {4, 8, 16}; otherwise {8, 16, 32}.* Result: all four policies had 0/20 at σ=32 → **{4, 8, 16}**; the "strongest noise level" used in H2 is therefore **σ = 16**. k=10 is not in the grid (0–1 successes in 20 episodes in every pilot condition).

## 4. Seeds

Final evaluation seeds **200000–200199**: 200 episodes per cell, **the same 200 seeds in every cell** (a seed fixes the initial scene, so outcomes of two policies on the same seed are paired). Seeds 0–9999 were used for debugging and the pilot (pilot: 5000–5019); 10000–199999 are reserved and unused.

## 5. Primary hypotheses (confirmatory)

Primary outcome: **episode success** (binary). Three hypotheses, each a **two-sided** test at α = 0.05, **corrected together with Holm** (family of exactly 3):

- **H1.** Under push 0.06 m, the success rate of k = 20 is higher than that of k = 100.
- **H2.** Under the strongest noise level (σ = 16), the success rate of k = 100 is higher than that of temporal ensembling (te).
- **H3.** Under action delay of 10 steps, the success rate of k = 100 is higher than that of k = 50.

**Test.** For each hypothesis, outcomes of the two policies on the 200 shared seeds are paired. Let b = number of seeds where only A (the first-named policy) succeeds and c = number where only B succeeds. The **exact McNemar test** is used: under H0, b ~ Binomial(b + c, 1/2); two-sided p = min(1, 2·P(X ≤ min(b, c))); p = 1 if b + c = 0.

**Effect size.** Report the success-rate difference (A − B) with a **paired bootstrap 95% confidence interval**: resample the 200 seeds with replacement 10,000 times (the same resampled seeds for A and B), take the 2.5th and 97.5th percentiles of the resampled differences; fixed RNG seed 20261007.

**Holm correction.** The three raw p-values are sorted ascending; the i-th smallest is multiplied by (3 − i + 1), the running maximum is taken, and values are capped at 1. A hypothesis is **supported** if its Holm-adjusted p < 0.05 and the difference is in the hypothesised direction. If the Holm-adjusted p < 0.05 and the difference is in the **opposite** direction, the result is recorded as an **"opposite result"** and reported as such. Otherwise the hypothesis is reported as not supported (not significant); no further inference is drawn.

## 6. Secondary outcomes and exploratory analyses

**Secondary outcomes (descriptive only, no tests):** for every cell, the mean over all episodes of (i) number of policy calls per episode, (ii) jerk, (iii) path_length, (iv) smoothness (definitions in `src/chunkre/metrics.py`; computed on the actions actually executed), each with a bootstrap 95% confidence interval (10,000 resamples over episodes, RNG seed 20261007). Episodes where a metric is undefined (e.g. too few steps) are left out of that metric's mean, and the number of valid episodes is reported.

**Exploratory analyses (explicitly labelled exploratory; no corrections, not evidence):**
1. Success-rate curves over perturbation strength for all 40 cells (with Wilson 95% intervals).
2. For each of the 10 conditions, which policy has the highest success rate (ties listed).

## 7. Basis in the pilot

The hypotheses come from a **pilot** (T6a + supplementary σ=32 run) on aloha ACT with **tuning seeds 5000–5019** (20 episodes per cell), results in `results/pilot/` (`pilot_table.md`, `pilot_table.csv`, `fig_*.png`). The pilot showed that the k = 20 / 50 / 100 / te policies separate under these perturbations (e.g. under push 0.06 m: k=20 11/20, k=100 3/20; under σ=16: k=100 15/20, te 10/20; under delay 10: k=100 11/20, k=50 6/20) and fixed the levels above. These are pilot numbers on 20 episodes and are **not** evidence; the final evaluation uses entirely new seeds (200000–200199).

## 8. Data rules

- **All episodes count; none is excluded.** For push cells, episodes in which the push was *skipped* (cube already lifted at the push step) or *truncated* (push distance shortened by the table edge or arm contact), or in which the episode ended before the push step, are included as they are, and their numbers are reported separately per cell.
- If a run is interrupted it is **resumed**; batches already completed are not re-run.
- The analysis is run once all 40 cells are complete; runs of the analysis script on incomplete data are only completeness checks and no design decision is based on them.
- Any deviation from this registration is written in the **"Deviations" ("偏离") section of the README**.

## 9. Further registrations (not part of this file)

Further pre-registrations will be added in separate files before the corresponding data are collected: (a) an adaptive re-observation rule (before day 11); (b) a Diffusion Policy replication; (c) a self-trained ACT replication. Nothing in this file depends on them.

---

## 中文摘要

**问题：** ACT 一次预测一串动作，执行前 k 步再重新观察。扰动（推方块、观测噪声、动作延迟）下，k 取多大、用不用时间集成（te），成功率怎么变？只测量，不声称新方法。

**设置：** 模型 `lerobot/act_aloha_sim_transfer_cube_human`，LeRobot 0.3.3，GPU 渲染（D3D12），B=10。**网格 4 策略 × 10 条件 = 40 格**：策略 k=20/50/100（fixed）+ te（系数 0.01）；条件 = 无扰动 + 推方块 0.02/0.04/0.06 m（窗口 [20,100]）+ 观测噪声 σ=4/8/16 + 动作延迟 3/5/10 步。噪声档位按事先定好的规则选定：σ=32 补测时 4 个策略都是 0/20（≤2/20），所以用 {4,8,16}，噪声最强档 = 16。

**种子：** 最终评估 200000–200199，每格 200 回合，所有格子同一批种子（按种子配对）；调参/试跑种子是 5000–5019，不重复使用。

**三个主要假设**（成功率，双侧，α=0.05，三者一起做 Holm 校正）：H1 推 0.06 m 时 k=20 > k=100；H2 噪声 σ=16 时 k=100 > te；H3 延迟 10 步时 k=100 > k=50。检验用按种子配对的 McNemar 精确检验，同时报告成功率差与配对 bootstrap 95% 区间（10000 次，随机种子 20261007）。方向相反且校正后显著的，记为“相反结果”，如实报告。

**次要指标**（策略调用次数、jerk、path_length、smoothness）只报告均值和 bootstrap 区间，不检验。**探索性分析**（已标注）：40 格成功率曲线、每种扰动下哪个策略最好。

**数据规则：** 所有回合都计入、不剔除；推方块的“跳过”“截断”回合照常计入并单独报告数量；中断就续跑、不重跑已完成批次；任何偏离写进 README 的“偏离”小节。后续会另外追加登记：自适应重新观察规则（第 11 天前）、Diffusion Policy 重复实验、自训练 ACT 重复实验。
