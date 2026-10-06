# T2 正确性检查结果（scripts/t2_checks.py analyze 自动生成）

## ACT（GPU 渲染 D3D12）
- V1 未通过：act_v1_run1 vs act_v1_run2，10 行，完全一致=False，不一致列={'success': 1, 'steps': 6, 'sum_reward': 10, 'max_reward': 1, 'n_policy_calls': 1, 'smoothness': 10}，奖励最大绝对差=182
- V2 未通过：act_v1_run1 vs act_v2_k_explicit，10 行，完全一致=False，不一致列={'success': 1, 'steps': 6, 'sum_reward': 10, 'max_reward': 1, 'n_policy_calls': 1, 'smoothness': 10}，奖励最大绝对差=80
- V3 未通过：缺结果
- V4 通过：
  - k=1（act_v4_k1）：平均调用 400.00，平均步数/k 400.00，调用=ceil(步数/k) 10/10，成功 0/10，截断 0，平均平滑度 0.001422
  - k=10（act_v4_k10）：平均调用 38.70，平均步数/k 38.45，调用=ceil(步数/k) 10/10，成功 4/10，截断 0，平均平滑度 0.02269
  - k=100（act_v1_run1）：平均调用 3.50，平均步数/k 3.22，调用=ceil(步数/k) 10/10，成功 7/10，截断 0，平均平滑度 0.02363
  - 三个 k 下逐回合（成功,步数,平滑度）完全相同的回合数：0/10（不应三者全同：满足）
  - te：跑通 1 批，成功 8/10，调用=步数 10/10

## ACT（CPU 渲染 llvmpipe，只做 V1–V3）
- V1 通过：act_cpu_v1_run1 vs act_cpu_v1_run2，10 行，完全一致=True，不一致列={}，奖励最大绝对差=0
- V2 通过：act_cpu_v1_run1 vs act_cpu_v2_k_explicit，10 行，完全一致=True，不一致列={}，奖励最大绝对差=0
- V3 通过：成功一致 10/10，步数一致 10/10，两者都一致 10/10；成功率 ours 0.90 / lerobot 0.90；sum_reward 最大差 0，max_reward 最大差 0

## DP
- V1 通过：dp_v1_run1 vs dp_v1_run2，10 行，完全一致=False，不一致列={'sum_reward': 1}，奖励最大绝对差=2.84e-14
- V2 通过：dp_v1_run1 vs dp_v2_k_explicit，10 行，完全一致=False，不一致列={'sum_reward': 1}，奖励最大绝对差=2.84e-14
- V3 通过：成功一致 10/10，步数一致 10/10，两者都一致 10/10；成功率 ours 0.50 / lerobot 0.50；sum_reward 最大差 0.4915，max_reward 最大差 0
- V4 通过：
  - k=1（dp_v4_k1）：平均调用 30.00，平均步数/k 30.00，调用=ceil(步数/k) 10/10，成功 0/10，截断 10，平均平滑度 12.7
  - k=8（dp_v1_run1）：平均调用 30.20，平均步数/k 29.77，调用=ceil(步数/k) 10/10，成功 5/10，截断 0，平均平滑度 9.695
  - k=15（dp_v4_k15）：平均调用 15.30，平均步数/k 14.91，调用=ceil(步数/k) 10/10，成功 7/10，截断 0，平均平滑度 9.812
  - 三个 k 下逐回合（成功,步数,平滑度）完全相同的回合数：0/10（不应三者全同：满足）

summary 可由 CSV 重算：14/14 个目录一致 {'act_cpu_v1_run1': True, 'act_cpu_v1_run2': True, 'act_cpu_v2_k_explicit': True, 'act_v1_run1': True, 'act_v1_run2': True, 'act_v2_k_explicit': True, 'act_v4_k1': True, 'act_v4_k10': True, 'act_v4_te': True, 'dp_v1_run1': True, 'dp_v1_run2': True, 'dp_v2_k_explicit': True, 'dp_v4_k1': True, 'dp_v4_k15': True}
