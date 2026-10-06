"""chunkre：动作块执行长度 k 研究的统一评估包（T2）。

- seed_guard：种子区间硬检查
- config：评估配置、k 上限校验
- metrics：平滑度、Wilson 区间、由逐回合 CSV 重算汇总（纯计算，可单测）
- runner：批量环境逐步循环、断点续跑、结果文件（需要 lerobot / 仿真）
"""
