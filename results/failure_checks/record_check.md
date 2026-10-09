# T13 指尖记录前后动作逐位相同检查（CPU 渲染，种子 0–4，B=5）

渲染器：`llvmpipe (LLVM 20.1.2, 256 bits)`。旧代码 = 加指尖记录之前的提交（git archive），新代码 = 当前工作区；两边都开 `--log-traj`。

| 条件 | 回合 | 总步数 | 成功 旧/新 | CSV 逐列相同（除墙钟） | 轨迹数组逐字节相同 | 新增数组 | 通过 |
|---|---|---|---|---|---|---|---|
| k20__none | 5 | 1520 | 5/5 | 是（31 列） | 是（10 个：action, agent_pos, alive, chunk_idx, chunk_offsets, chunk_sel, cube_pos, repredict, seeds, steps） | finger_tip | 通过 |
| k20__push0.06 | 5 | 1901 | 2/2 | 是（31 列） | 是（11 个：action, agent_pos, alive, chunk_idx, chunk_offsets, chunk_sel, cube_pos, cube_pos_after, repredict, seeds, steps） | finger_tip | 通过 |
| k20__delay10 | 5 | 2000 | 0/0 | 是（31 列） | 是（11 个：action, agent_pos, alive, chunk_idx, chunk_offsets, chunk_sel, cube_pos, policy_action, repredict, seeds, steps） | finger_tip | 通过 |

**结论：全部通过**。
