# T3：ACT k=1 诊断结论

**结论（三选一）：策略的真实行为。** 评估脚本在 k=1 下是忠实的；ACT 在“每步重新预测、只执行新块第 1 个动作、不做时间集成”时，手臂在约 45 步内停在起始姿态附近，此后原地微动，不去抓方块。

数字全部来自本目录（`diag.json` / `diag.md` 由 `python scripts/t3_diag.py analyze` 从 episodes.csv 和 traj/*.npz 生成）。

## 证据
1. **Q1 脚本忠实**（CPU 渲染 llvmpipe，逐字节确定；种子 0–9，B=10，k=1）
   - lerobot 0.3.3 自带 `eval_main`（策略由 lerobot 自己按 `--policy.n_action_steps=1` 加载）与我们的脚本逐回合（成功, 步数）一致 **10/10**；两边都是 0/10 成功、每回合跑满 400 步。
   - 因为全失败时这个一致很容易满足，又加了逐步比较：lerobot（`--env.episode_length=100`）实际发给环境的动作与我们 `--log-traj` 记录的动作，**10 个环境 × 前 100 步逐位相同**（最大差 0）。
   - `--log-traj --save-frames 0` 开 / 关：逐回合（除墙钟列）**10/10 完全一致**。
2. **Q2 剂量反应**（GPU 渲染，种子 0–19，B=10）：成功 k=1 0/20、k=2 0/20、k=5 0/20、k=10 6/20、k=100 17/20、te 18/20；path_length 0.64 / 1.61 / 3.78 / 8.09 / 7.35 / 6.19；方块最大抬升 0.000 / 0.000 / 0.038 / 0.143 / 0.264 / 0.267 m；有夹爪闭合的回合 0 / 1 / 4 / 12 / 20 / 20。执行长度越短，手臂走得越少、越抓不到。
3. **Q3 停滞而不是大幅来回抖**（12 个手臂关节，弧度）：新预测块的第 1 / 5 / 10 / 20 个动作与当时关节位置的平均距离，k=1 为 0.015 / 0.022 / 0.053 / 0.210，k=100 为 0.076 / 0.097 / 0.130 / 0.200。即策略每次给出的计划在第 20 步都要走约 0.2 rad（两种 k 下差不多），但计划的开头几乎原地不动；k=1 只执行开头，所以永远在“起步”。
   - k=1 的手臂在第 44.5 步（20 回合中位数）之后一直在最终位置 0.01 rad 以内；第 200 步以后每步实际移动 6.3e-4 rad（k=100 为 1.4e-2，约小 22 倍）。
   - k=1 的 reversal_rate 0.56 很高，但对应的是这 6e-4 rad 量级的原地微动（静止点附近的数值抖动），不是大幅来回摆动。
4. **Q4 画面**：`q2_k1/frames/seed0.gif`（第 390 步双臂仍在起始姿态附近，方块未动）；`q2_k100/frames/seed0.gif`（第 150 步右臂已夹住方块）。

## 文件
- `q1_ours_cpu_k1/`、`q1_ours_cpu_k1_traj/`（我们的脚本，--log-traj 关 / 开）；`q1_lerobot_cpu_k1/`（lerobot eval_main 全长）；`q1_lerobot_cpu_k1_len100/`（lerobot 前 100 步 + actions.npz）。
- `q2_<k1|k2|k5|k10|k100|te>/`：config.json、episodes.csv、summary.json、traj/batch_000{0,1}.npz；k1、k100 另有 frames/。
- `diag.json` / `diag.md`：Q1–Q4 汇总；`summarize_check.txt`：22 个结果目录 summary 均可由 CSV 重算；`verify_env_from_root.txt`：从 /root 调用 verify_env.sh。
- 第一次 lerobot 运行因关掉视频后 eval_main 不建输出目录而在写 eval_info.json 时失败（`q1_lerobot_attempt1_failed.log`，已修复并重跑；那次打印的 pc_success 也是 0.0）。
