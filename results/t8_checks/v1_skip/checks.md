# T8b 检查 D1–D3（PushT 推 T 形块，DP）

## D1 强度 0 = 无扰动 —— 通过

- results/t8_dp_baseline/run_k8（不开扰动）vs results/t8_checks/d1_d0（push d=0），种子 0–49，B=50，k=8
- 逐回合一致 50/50（除墙钟列、扰动列外，其他列字符串完全相同；sum_reward、max_reward 允许 ≤ 1e-09）；超出的列：无
- 奖励列：sum_reward 最大差 9.9e-14（不逐位相等 6 回合），max_reward 最大差 2.2e-16（不逐位相等 5 回合）
- d=0 运行的 push_status：['zero']；整回合执行动作逐位相同：50/50；成功 33/50 vs 33/50
- 对照（同一无扰动配置重跑，results/t8_checks/d1_none_rerun）：逐回合一致 50/50，整回合动作逐位相同 50/50；sum_reward 最大差 8.5e-14（不逐位相等 6 回合），max_reward 最大差 1.1e-16（不逐位相等 4 回合）。即奖励列的 1 ulp 级差异在不开扰动时也有（PushT 环境自身的浮点不确定性，来源未查），与扰动钩子无关。

## D2 扰动生效（d=60，k=8，种子 0–49）—— 通过

- 推了 33（其中撞墙截断 2），skipped_overlap 17，not_reached 0（共 50）
- 推的那一步位移长度与期望（未截断 = d，截断 = push_d_actual）的最大误差 2.33e-05（轨迹为 float32）；最大转角变化 0；其他步 after−before 最大 0
- 不合格种子：无
- 推前推后对比图：`results/t8_checks/d2_seed0_step25.png`、`results/t8_checks/d2_seed1_step59.png`

### 三档 d 的截断 / 跳过数（k=8，用 D2 轨迹离线重放；d=最大档与实际运行逐回合一致：是）

| d | skipped_overlap | 撞墙截断 | 推的时候智能体正贴着 T | 其中被跳过 |
|---|---|---|---|---|
| 20 | 17/50 | 1 | 25 | 15 |
| 40 | 18/50 | 2 | 25 | 12 |
| 60 | 17/50 | 2 | 25 | 11 |

## D3 粗看（只看方向，不作结论）

| 格子 | 成功 | 成功率 [Wilson 95%] | 平均 max_reward | 推的状态 |
|---|---|---|---|---|
| k4_none | 14/50 | 0.28 [0.17, 0.42] | 0.946 | {'none': 50} |
| k4_dmax | 9/50 | 0.18 [0.10, 0.31] | 0.907 | {'pushed': 31, 'skipped_overlap': 19} |
| k8_none | 33/50 | 0.66 [0.52, 0.78] | 0.926 | {'none': 50} |
| k8_dmax | 27/50 | 0.54 [0.40, 0.67] | 0.923 | {'pushed': 33, 'skipped_overlap': 17} |
| k15_none | 25/50 | 0.50 [0.37, 0.63] | 0.909 | {'none': 50} |
| k15_dmax | 26/50 | 0.52 [0.39, 0.65] | 0.955 | {'pushed': 29, 'skipped_overlap': 21} |
