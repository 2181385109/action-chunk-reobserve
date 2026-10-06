# 相关文献核实（T1，2026-10-06）

核实方式：2026-10-06 逐篇访问 arXiv 摘要页（arxiv.org/abs/<号>）及会议/期刊官方页面。**只写页面上看得到的信息**；页面没写的标“未说明”，没核实到的标“未核实”。

| # | 简称 | 标题（arXiv 页面） | 作者（全名） | 年份 | 发表场所 | arXiv | 核实状态 |
|---|---|---|---|---|---|---|---|
| 1 | ACT | Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn | 2023（首次提交 2023-04-23） | RSS 2023（Robotics: Science and Systems XIX，2023 年 7 月，韩国大邱；RSS 官方论文集页 roboticsproceedings.org/rss19/p016.html 核实） | 2304.13705 | **已核实**（arXiv 页面本身未标注 venue，venue 来自 RSS 官方页） |
| 2 | Diffusion Policy | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, Shuran Song | 2023（首次提交 2023-03-07） | RSS 2023（roboticsproceedings.org/rss19/p026.html）；扩展期刊版：International Journal of Robotics Research, Vol. 44, No. 10-11, pp. 1684–1704 | 2303.04137 | **已核实**（arXiv 页面标注“Extended journal version of the original RSS2023 paper”；IJRR 卷期页码来自 ACM DL 搜索结果页，未打开期刊原页） |
| 3 | Bidirectional Decoding (BID) | arXiv v1 标题：Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling；ICLR 页面标题：…via Closed-Loop Resampling（标题在版本间改过） | Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Maximilian Du, Chelsea Finn（ICLR 页面署名写作 Jubayer Hamid, Max Du，为简写） | 2024（首次提交 2024-08-30；最新 v4 为 2025-04-25） | ICLR 2025 —— 但 iclr.cc 页面（iclr.cc/virtual/2025/32377）把它列为 **Workshop “Generative Models for Robot Learning” 的 Poster**；是否同时是主会论文：**未核实**。arXiv 页面未标 venue | 2408.17355 | **部分核实**（作者、年份、arXiv 号已核实；“主会还是 workshop”未核实，引用时写 “ICLR 2025 (workshop poster listed on iclr.cc)” 或只写 arXiv） |
| 4 | Real-Time Chunking (RTC) | Real-Time Execution of Action Chunking Flow Policies | Kevin Black, Manuel Y. Galliker, Sergey Levine | 2025（首次提交 2025-06-09） | NeurIPS 2025（来自 arXiv 摘要页的 venue 字段；NeurIPS 官方页未单独打开） | 2506.07339 | **已核实**（arXiv 页面）；NeurIPS 官方页：**未核实** |

> 注：提示词里写“Real-Time Chunking（Black 等，2025）”，论文正式标题是 “Real-Time Execution of Action Chunking Flow Policies”，“Real-Time Chunking (RTC)” 是其方法名。

## 与“执行多少步再重新观察”的关系

> 说明：下面各段对论文内容的概括，除 Diffusion Policy 的“receding horizon control”、BID 的 backward coherence / forward contrast、RTC 的“异步执行、块边界连续”来自已打开的摘要/搜索结果外，其余（如 ACT 的时间集成细节、RTC 的 inpainting 机制）来自背景知识，**未逐段对照原文**；写论文引用前需要对原文再核一次。

**ACT（Zhao 等，2023）**
ACT 提出“动作分块”：策略一次预测未来 k 步动作，以降低复合误差、缓解任务有效时间跨度。论文里同时提出“时间集成”（temporal ensembling）：每一步都重新预测，再对重叠的多个预测做加权平均，相当于“每步都重新观察”（k=1）并用平滑换取一致性。这直接给出了 k 的两个极端：开环执行整个块 vs 每步重算并集成。本项目把 k 作为受控变量，并在扰动下测它与成功率/平滑度/算力的取舍。

**Diffusion Policy（Chi 等，2023）**
Diffusion Policy 用扩散模型生成动作序列，并明确采用“滚动时域控制”（receding horizon control）：预测一个长的动作序列，只执行前面一部分（执行长度），然后重新观察、重新预测。论文把预测长度与执行长度区分开，并讨论了执行长度过长会降低反应性、过短会导致动作不连贯（多模态下来回切换）。这正是本项目要在扰动下系统测量的变量。

**Bidirectional Decoding（Liu 等，2024）**
BID 指出动作分块一方面更好地捕捉演示中的时间依赖，另一方面在随机环境中降低反应性，并提出一种测试时采样方法：每步采多个候选，按“与上一步决定一致”（backward coherence）和“对未来更有利”（forward contrast）选择，把分块与闭环桥接起来。它是“靠更聪明的推理算法解决 k 的取舍”，而本项目只做 k 的受控对照测量，不提出新方法；BID 可作为将来对照的背景，不在本项目范围内实现。

**Real-Time Chunking（Black 等，2025）**
RTC 面向流匹配/扩散类动作分块策略的异步实时执行：在当前块执行期间就开始生成下一块，通过“修补”（inpainting）式的约束保证块边界处动作连续。它处理的是“推理有延迟时，块与块如何衔接”，与本项目的“动作延迟扰动”和“块边界平滑度”相关；但它是新推理算法，本项目不实现它，只在讨论部分把它作为相关工作引用。
