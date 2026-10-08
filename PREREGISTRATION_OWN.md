# Pre-registration (OWN): do the ACT findings on execution length hold for a self-trained ACT model?

Registered on 2026-10-08, **before any evaluation of the self-trained model has been run** (this includes the base-level test in §3; the only evaluation of any self-trained checkpoint so far is a 2-episode format smoke test, seeds 0–1, of the 200-step *smoke* checkpoint made in T10a, a different checkpoint from the one registered here). Final seeds >= 200000 are blocked for locally trained ACT checkpoints by `src/chunkre/seed_guard.py` until this file is committed. Repository: `2181385109/chunk-reobserve`. This file is separate from `PREREGISTRATION.md` (ACT, commit `6c5302c`) and `PREREGISTRATION_DP.md` (DP, commit `e963f3f`, wording `300bf15`); both are **not modified**. The ACT family H1–H3 and the DP family H4–H5 are unchanged and are not affected by this file.

## 1. Purpose

`PREREGISTRATION.md` tested the pretrained hub model `lerobot/act_aloha_sim_transfer_cube_human`. That is one training run by someone else. This file registers a **replication on an ACT model that we trained ourselves** with the official recipe (one independent training run, one seed), to check whether two of the ACT conclusions are properties of the execution length *k* rather than of one particular set of weights: H1 → **H1'** (short k is better under a pushed cube), H3 → **H3'** (long k is better under action delay). It is a controlled measurement study; no new method is claimed. The pretrained results for H1 and H3 were already known when this file was written (H1 supported, H3 supported, `results/main/`), so H1'/H3' are replications chosen *because* those two held, not new discoveries; H2 (observation noise, not significant for the pretrained model) and temporal ensembling are not replicated here (cost; no confirmatory result to replicate).

## 2. Model (fixed, chosen before any evaluation result exists)

| Item | Value |
|---|---|
| Model | self-trained ACT, **final checkpoint, step 100000** |
| Path (WSL) | `/root/train_out/act_own/checkpoints/100000/pretrained_model` (`checkpoints/last` points to the same directory) |
| `model.safetensors` SHA-256 | `28631d3abc89d054d5d26460bf604377db3c3f054138c4e3c588cd6bd6bda4d1` |
| `config.json` / `train_config.json` SHA-256 | `2467318d078d2feaeb755e9c87c782049b1f2dd3819e96a7825e6231294357b8` / `467e01fee9bfe5a117121a7b59790251fd1e8a33a6ef6b8209cc95189e27eded` |
| Training | `python -m lerobot.scripts.train` of LeRobot 0.3.3 (pinned), dataset `lerobot/aloha_sim_transfer_cube_human` (50 episodes, 20000 frames), **seed 1000**, batch size 8, lr 1e-5, weight decay 1e-4, gradient clip 10, 4 data-loader workers, **all policy hyperparameters as in the official `train_config.json`** (`docs/official_model_cards/…_train_config.json`); the deviations from that file (steps/save frequency/logging/evaluation off/no W&B/no hub push/output dir/video backend) are listed in `results/train_act_own/prep/train_config_diff.json`. |
| Steps | 100000 (the official step count; chosen by the rule "official steps if the estimated run time is ≤ 10 h", estimate 4.76 h; see `HANDOFF.md`). One uninterrupted run (no resume; `results/train_act_own/train_log.txt`, one `TRAIN_ACT_START` line). |
| Video decoding | torchcodec 0.5 (the official recipe used pyav; on this machine the pyav path fails; decoded pixels were checked to be identical to PyAV on sample frames, T10a). |
| Known differences from the official model | other weights (one new training run); LeRobot 0.3.3 instead of the 2025-01 commit used for the official run; torchcodec decoding; the official hub model is the step-80000 snapshot of its run while ours is step 100000. |

**Checkpoint rule.** The final checkpoint (step 100000) was fixed as *the* evaluated checkpoint **before any evaluation result of any of the 20 saved checkpoints (steps 5000–100000) existed**. No checkpoint will be selected, compared, or replaced on the basis of evaluation results, and the other 19 checkpoints will not be evaluated for this experiment. The training loss curve (`results/train_act_own/loss.csv`) has been seen; it is not used for selection (the step count was fixed by rule before training).

## 3. Base-level test (descriptive only)

The self-trained model is run with **k = 100** (`mode: fixed`, no perturbation), seeds **1000–1499 (500 episodes)**, B = 10, GPU rendering — the same protocol as the reproduction of the pretrained model (`results/repro/act`). Reported: number of successes, success rate, Wilson 95% interval. It is listed next to the **official 83.0%** and our **reproduction of the pretrained model 82.6% (413/500, Wilson [79.0, 85.7])**. **No test is performed** and no hypothesis is attached to this number.

**Weak-model rule (stated in advance).** If the success rate of the base-level test is **below 40%**, the main experiment is run as planned, but the README states that *"the self-trained model is weak; the replication conclusions are of limited reference value"*. The rule changes nothing else (no cell is dropped, no hypothesis is changed).

## 4. Main experiment: 3 values of k × 3 conditions = 9 cells

Setup identical to `PREREGISTRATION.md` §2 (LeRobot 0.3.3; GPU rendering EGL + Mesa D3D12, the evaluation script aborts if the renderer is not D3D12; B = 10; success = `is_success` at the end of the 400-step episode; ACT under D3D12 rendering is reproducible only statistically). Evaluation and perturbation logic as of commit `4b6aadf` for ACT (T10a, `7880cb9`, added only the ability to load a local checkpoint via `--policy-path`; T10b adds the seed guard extension, run and analysis scripts and no evaluation logic).

**Execution length (3):** k ∈ {20, 50, 100} (`mode: fixed`). **Conditions (3), definitions exactly as in `PREREGISTRATION.md` §3:**

| Condition | Cell tag | Definition |
|---|---|---|
| none | `none` | no perturbation |
| push cube 0.06 m | `push0.06` | at a uniformly random step in the window [20, 100], in a uniformly random horizontal direction, the cube is displaced by 0.06 m (per-episode RNG derived from the seed; `PushCube`) |
| action delay 10 steps | `delay10` | the action executed at step t is the policy output of step t−10 (the first 10 steps hold the initial joint positions); 50 Hz, i.e. 200 ms |

Cell names: `<k>__<condition>`, e.g. `k20__push0.06`, `k100__delay10`, `k50__none`; outputs in `results/main_own/<cell>/`. Observation noise and temporal ensembling are **not** in this grid.

**Seeds.** Final evaluation seeds **200000–200199**, 200 episodes per cell, **the same 200 seeds in every cell** (outcomes of two cells on the same seed are paired). These are the same seeds as in the pretrained-model experiment (same task and environment, so the initial scenes coincide; this makes a same-seed comparison between the two models possible, §6). Seeds 0–9999 are for debugging, 10000–199999 reserved and unused.

## 5. Primary hypotheses (confirmatory)

Primary outcome: **episode success** (binary). Two hypotheses, each a **two-sided** test at α = 0.05, **corrected together with Holm** (family of exactly 2; separate from, and not combined with, the ACT family H1–H3 or the DP family H4–H5):

- **H1'** (does ACT H1 hold for the self-trained model?). Under push 0.06 m, the success rate of k = 20 is higher than that of k = 100.
- **H3'** (does ACT H3 hold for the self-trained model?). Under action delay of 10 steps, the success rate of k = 100 is higher than that of k = 50.

**Test, effect size, Holm, "opposite result": identical to `PREREGISTRATION.md` §5.** Outcomes of the two cells on the 200 shared seeds are paired; b = seeds where only A (first-named) succeeds, c = seeds where only B succeeds; exact McNemar test (b ~ Binomial(b + c, 1/2) under H0; two-sided p = min(1, 2·P(X ≤ min(b, c))); p = 1 if b + c = 0). Effect size = success-rate difference (A − B) with a paired bootstrap 95% CI (resample the 200 seeds with replacement 10,000 times, the same resampled seeds for A and B, 2.5th/97.5th percentiles, fixed RNG seed 20261007). Holm: sort the 2 raw p-values, multiply the i-th smallest by (2 − i + 1), take the running maximum, cap at 1. A hypothesis is **supported** if its Holm-adjusted p < 0.05 and the difference is in the hypothesised direction; if Holm-adjusted p < 0.05 in the **opposite** direction it is recorded as an **"opposite result"** and reported as such; otherwise it is reported as not supported (not significant) and no further inference is drawn.

**Stated in advance.** A weak self-trained model can make a hypothesis untestable in practice (floor effect: e.g. if k = 50 and k = 100 both succeed almost never under delay 10, H3' will be non-significant). That would be reported as non-significant, not as evidence against the ACT finding.

## 6. Secondary outcomes and exploratory analyses

**Secondary outcomes (descriptive only, no tests):** as in `PREREGISTRATION.md` §6, for every cell the mean over all episodes of (i) policy calls per episode, (ii) jerk, (iii) path_length, (iv) smoothness (definitions in `src/chunkre/metrics.py`), each with a bootstrap 95% CI (10,000 resamples over episodes, RNG seed 20261007; undefined episodes left out, valid number reported). For push cells the numbers of pushed / skipped / truncated episodes are reported.

**Exploratory analyses (explicitly labelled exploratory; no corrections, no significance tests, not evidence):**

1. **Gap change**, exactly as defined in `PREREGISTRATION_DP.md` §6.2, for the self-trained model: S = k 20, L = k 100, C ∈ {push 0.06 m, delay 10}; per seed *i*, `gap_change_i = [s_S(i,C) − s_L(i,C)] − [s_S(i,none) − s_L(i,none)]`, mean over the seeds with a paired bootstrap 95% CI (10,000 resamples, RNG seed 20261007, the same resampled seeds in all four cells). The pretrained model's gap change on the same conditions is shown next to it for comparison.
2. **Pretrained vs. self-trained agreement on the same seeds** (descriptive only): for each of the 9 cells (pretrained data `results/main/`, self-trained `results/main_own/`, same seeds 200000–200199) and for the base-level test (`results/repro/act` vs `results/own_baseline/`, seeds 1000–1499), the 2×2 table of outcomes and the fraction of seeds on which the two models both succeed or both fail. No test. (Outcomes of a model on a seed are not noise-free under D3D12 rendering, so even a model compared with itself would not agree 100%.)
3. Success-rate table and curves for the 9 cells (Wilson 95% intervals); which k has the highest success rate in each condition (ties listed).

## 7. Basis

The design is the ACT design of `PREREGISTRATION.md` restricted to the three conditions that carry H1 and H3 (+ the no-perturbation reference needed for the gap change). Results for the pretrained model on these 9 cells are in `results/main/` (known): e.g. push 0.06: k=20 102/200, k=100 37/200; delay 10: k=100 95/200, k=50 39/200, k=20 0/200. They are **not** evidence about the self-trained model (a new model is evaluated, on the same seeds as the pretrained experiment, §4). No pilot with the self-trained model exists or will be run; the hypotheses and levels were not tuned on any evaluation of it.

## 8. Data rules

Identical to `PREREGISTRATION.md` §8:

- **All episodes count; none is excluded.** Push episodes that were skipped, truncated or ended before the push step are included as they are, and their numbers are reported per cell.
- An interrupted run is **resumed**; batches already completed are not re-run. The checkpoint path and hash are written to each cell's `config.json`; a cell with a different checkpoint cannot be resumed into the same directory.
- The analysis is run once the base-level test and all 9 cells are complete; runs of the analysis script on incomplete data or on the pretrained-model data (dry runs) are only completeness / code checks, and no design decision is based on them.
- Any deviation from this registration is written in the **"Deviations" ("偏离") section of the README**.

## 9. Further registrations (not part of this file)

An adaptive re-observation rule will be registered in a separate file before the corresponding data are collected. Nothing in this file depends on it.

---

## 中文摘要

**目的：** 预训练模型只是别人的一次训练结果。这里检验 ACT 的两条结论（H1 推方块时短 k 更好、H3 动作延迟时长 k 更好）在**我们自己按官方配方训练的 ACT** 上是否仍成立。只测量，不声称新方法。与 `PREREGISTRATION.md`（ACT）和 `PREREGISTRATION_DP.md`（DP）完全独立，两份文件不改动，各自的 Holm 校正族不受影响。H1、H3 在预训练模型上已经成立（已知），所以 H1'、H3' 是“为复现而挑的”，不是新发现；H2（噪声，预训练模型上不显著）和时间集成不在这里复现。

**模型：** 自训练 ACT，**第 100000 步（最终）检查点**，路径 `/root/train_out/act_own/checkpoints/100000/pretrained_model`，`model.safetensors` 的 SHA-256 已写入；训练用 LeRobot 0.3.3，数据集 `lerobot/aloha_sim_transfer_cube_human`，种子 1000，官方超参数，torchcodec 解码，一次不间断训练。**检查点在看到任何评估结果之前就选定为最终检查点，不做挑选**，其余 19 个检查点不评估。

**基础水平测试（只描述）：** k=100、种子 1000–1499（500 回合），报告成功率和 Wilson 区间，与官方 83.0% 和我们复现的预训练模型 82.6% 并列，不做检验。**如果成功率低于 40%**，主实验照常进行，但在 README 注明“自训练模型较弱，重复结论的参考价值有限”。

**网格：** k ∈ {20, 50, 100}（fixed）× 条件 {无扰动；推方块 0.06 m；动作延迟 10 步} = 9 格，扰动设置与 `PREREGISTRATION.md` 完全相同；种子 200000–200199（与预训练模型实验同一批种子），每格 200 回合，按种子配对，B=10，GPU 渲染。

**两个主要假设**（成功率，双侧，α=0.05，两者一起做 Holm 校正，与其他登记分开）：**H1'** 推 0.06 m 时 k=20 的成功率高于 k=100；**H3'** 延迟 10 步时 k=100 的成功率高于 k=50。检验（按种子配对的 McNemar 精确检验）、bootstrap（10000 次，随机种子 20261007）、“相反结果”的定义与 `PREREGISTRATION.md` 相同。事先说明：模型较弱时可能出现地板效应（例如延迟 10 时 k=50、k=100 都几乎全败），那样 H3' 只会“不显著”，不能当作反对 ACT 结论的证据。

**次要指标**同 `PREREGISTRATION.md`（调用次数、jerk、path_length、smoothness，均值 + bootstrap 区间）。**探索性分析**（不检验、不是证据）：① 与 `PREREGISTRATION_DP.md` 相同定义的“差距变化”（k=20 对 k=100，条件为推 0.06 和延迟 10），并列出预训练模型的值；② 预训练模型与自训练模型在同一种子上的成败一致率（9 格 + 基础水平测试，只描述）；③ 9 格成功率表/曲线和每种条件下最好的 k。

**数据规则：** 与 `PREREGISTRATION.md` 相同——所有回合都计入、不剔除；中断就续跑；检查点路径和哈希记入每格 `config.json`；试跑数据上的分析脚本运行只是代码演示；任何偏离写进 README 的“偏离”小节。
