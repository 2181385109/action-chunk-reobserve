# 环境说明（T1，2026-10-06）

> 一句话：WSL2 Ubuntu-24.04 + Python 3.12 + **LeRobot 0.3.3** + PyTorch 2.7.1(cu126)，RTX 4060 Laptop 8GB；MuJoCo 渲染走 **EGL + Mesa D3D12（GPU）**；HF 直连、pip 走清华镜像。
> 所有 Python 在 WSL 里跑，入口 `source scripts/env.sh`（见 CLAUDE.md §3）。

## 1. 固定版本

| 项 | 版本 | 说明 |
|---|---|---|
| 系统 | WSL2 Ubuntu-24.04，root，系统 Python 3.12.3 | 宿主机驱动 592.27，WSL 里 `nvidia-smi` 可用 |
| venv | `/root/venvs/chunk-reobserve`（`python3 -m venv` + `uv pip`） | 系统原本没有 `python3.12-venv`/pip，用 apt 装（见 `setup_env.sh`） |
| **lerobot** | **0.3.3**（PyPI，正式版；extras `aloha,pusht`） | 选择理由见 §2 |
| torch / torchvision | 2.7.1+cu126 / 0.22.1（`torch.version.cuda`=12.6，cuDNN 9.5.1） | lerobot 0.3.3 要求 `torch<2.8`；`torch.cuda.is_available()==True` |
| gym-aloha / gym-pusht | 0.1.4 / 0.1.6 | 由 lerobot 的 `aloha`、`pusht` extras 安装 |
| gymnasium | 0.29.1 | lerobot 0.3.3 要求 `<1.0` |
| mujoco / dm-control | 3.8.1 / 1.0.41 | |
| numpy | 2.5.3 | |
| huggingface-hub | 1.33.0 | lerobot 0.3.3 只要求 `>=0.34.2`，实测可用 |
| diffusers | 0.41.0 | Diffusion Policy 需要 |
| 完整清单 | `requirements.lock.txt`（137 个包） | `setup_env.sh` 用它做约束，可重复运行（已验证：二次运行 `Checked 1 package`，无变化） |

安装：`wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/d/xiangmu/chunk-reobserve/scripts/setup_env.sh`
验证：`... scripts/verify_env.sh`，实测输出：
`lerobot 0.3.3 | torch 2.7.1+cu126 | cuda_available True | gpu NVIDIA GeForce RTX 4060 Laptop GPU`

## 2. 为什么是 0.3.3（版本选择）

条件：(a) 两个 hub 模型能用官方 API 直接加载、不需要手工转换；(b) aloha 与 pusht 环境能创建；取 PyPI 上满足的最新正式版。

PyPI 上的 lerobot 正式版：0.1.0、0.3.2、0.3.3、0.4.0–0.4.4、0.5.0、0.5.1、0.6.0、0.6.1（`>=0.5.0` 要求 Python>=3.12，系统 Python 3.12 都可用）。

两个 hub 模型是**旧格式**（2025-03 上传，仓库里只有 `config.json`/`model.safetensors`/`train_config.json`，**没有** `policy_preprocessor.json`，归一化统计量存在权重文件里）。实测：

| 版本 | 策略权重加载 | 归一化处理器加载（`make_pre_post_processors(cfg, pretrained_path=repo)`） | 结论 |
|---|---|---|---|
| 0.6.1 | 能 | **失败**：`ProcessorMigrationError: Model 'lerobot/act_aloha_sim_transfer_cube_human' requires migration to processor format. Run: python src/lerobot/processor/migrate_policy_normalization.py ...` | 需要手工转换 → 不满足 (a) |
| 0.4.4 | 能 | **失败**：`FileNotFoundError: Could not find 'policy_preprocessor.json' on the HuggingFace Hub at 'lerobot/act_aloha_sim_transfer_cube_human'`（DP 同） | 同上 |
| 0.4.0–0.4.3、0.5.0、0.5.1、0.6.0 | 未逐个测 | 源码中同样含 `migrate_policy_normalization.py` 迁移机制（0.4.4/0.5.1/0.6.1 的 wheel 已核对） | 视为同上 |
| **0.3.3** | 能 | 不需要（归一化在权重里） | **满足 (a)**；两个模型在环境里完整跑完回合（见 §6） |

> 注意：0.4.x 之后“策略权重能加载”不等于能正确推理——归一化统计量不在新处理器里，不转换就没有归一化。所以判据用的是“处理器能否加载”。

(b)：在 0.3.3 下 `AlohaEnv(task="AlohaTransferCube-v0")` 与 `PushtEnv()` 经 `make_env` 创建成功，并完成回合。候选版本检查脚本：`scripts/check_version_candidates.sh`。

不装 git 开发版，不改 LeRobot 源码。

## 3. 网络 / 镜像（机器在中国大陆，2026-10-06 实测，`scripts/netcheck.sh`）

| 站点 | 结果 | 决定 |
|---|---|---|
| huggingface.co | HTTP 200，~1 s；下载 ~6.1 MB/s | **直连**（`HF_ENDPOINT=https://huggingface.co`） |
| hf-mirror.com | HTTP 200，~0.9 s；下载 ~6.4 MB/s | 备用：`export HF_ENDPOINT=https://hf-mirror.com` |
| pypi.org | 15 s 超时 | 不用 |
| pypi.tuna.tsinghua.edu.cn | 200，3 s | **pip/uv 用清华镜像**（`PIP_INDEX_URL`、`UV_DEFAULT_INDEX`） |
| mirrors.aliyun.com/pypi | 15 s 超时 | 不用 |
| github.com | 200，0.8 s | 可达 |
| download.pytorch.org | 200 | ACT 首次加载时从这里下载 ResNet18 预训练权重（44.7 MB，已缓存到 `/root/.cache/torch`） |
| apt | `mirrors.aliyun.com/ubuntu`（系统原有配置） | 较慢（`python3-pip` 带出完整编译工具链，约 80 MB） |

torch 2.7.1 直接从清华镜像的 PyPI 轮子装（自带 CUDA 12.6 运行库），未使用 download.pytorch.org 的 cu 索引。

## 4. 渲染后端

aloha 是 MuJoCo 环境，需要离屏渲染。实测（`scripts/bench_render.sh`，aloha 零动作 100 步，480×640 相机）：

| 后端 | 实际渲染器（`GL_RENDERER`） | 每步耗时 |
|---|---|---|
| `MUJOCO_GL=egl`（默认） | Mesa **llvmpipe**（CPU 软渲染；WSL 里没有 NVIDIA EGL） | **67.0 ms/步** |
| `MUJOCO_GL=osmesa` | OSMesa 软渲染（CPU） | 66.0 ms/步（只测 20 步） |
| `MUJOCO_GL=egl` + `GALLIUM_DRIVER=d3d12` + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` | **D3D12 (NVIDIA GeForce RTX 4060 Laptop GPU)** | **11.6 ms/步** |

**最终选择：EGL + Mesa D3D12（GPU 渲染）**，已写进 `scripts/env.sh`（`CHUNK_RENDER=gpu` 为默认；`export CHUNK_RENDER=cpu` 退回 llvmpipe）。aloha 每回合环境耗时从 ~17.5 s 降到 ~4 s（约 4–5 倍）。PushT 用 pygame 渲染，不受影响。

⚠️ **渲染器是一个混杂因素，全研究必须固定同一种**：
- 同一个种子、同一个渲染器，结果逐步可复现（ACT seed 0 连跑 3 次，GPU 渲染 3 次都是 400 步失败，CPU 渲染 3 次都是 257 步成功）。
- 但两种渲染器的第一帧只有极小的像素差（平均绝对差 0.025/255，仅 0.007% 像素差 >8），ACT 的 seed 0 却从“成功”变成“失败”：开环执行 100 步的 ACT 会放大微小的图像差异。**单个种子的成败对渲染器很敏感**；这也说明：官方 83% 是在另一种硬件/渲染器上测得的，我们复现时数字有偏差是正常的（后续任务再测）。
- 退出时 stderr 会出现 `OpenGL.raw.EGL._errors.EGLError` / `render_executor ... atexit` 的追溯，是 dm_control 在解释器退出时释放 EGL 上下文的无害噪声，不影响结果，退出码为 0。

## 5. 官方基线（83% 与 65.4% 的来源与设置）

来源：两个 hub 模型卡 `README.md`、`eval_info.json`、`config.json`、`train_config.json`（2026-10-06 下载；README/config/train_config 的副本在 `docs/official_model_cards/`；hub 当前 main 版本 `ba73b276…`(ACT)、`84a7c231…`(DP)）。LeRobot 文档未另外核对其数字，下面以模型卡为准。

| 项 | `act_aloha_sim_transfer_cube_human` | `diffusion_pusht` |
|---|---|---|
| 官方数字 | **83.0%**（500 回合；对照“Theirs”=原 ACT 仓库权重在 LeRobot 评估 68.0%） | **65.4%**（500 回合；平均最大重叠 0.955；对照“Theirs”64.2%） |
| 成功定义 | 方块被一只机械臂拿起并交接到另一只（模型卡注：LeRobot 的 aloha 环境在第二只夹爪与方块“对向接触”时即记成功，可能偏宽松） | 与目标的最大重叠 ≥ 95% |
| 回合数 | 500 | 500 |
| 评估命令（模型卡原文给出） | `python lerobot/scripts/eval.py --policy.path=…/080000/pretrained_model --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=500 --eval.batch_size=50 --device=cuda --use_amp=false` | 模型卡给出了训练命令，评估命令**未说明**（`eval_info.json` 对应同样的 500 回合） |
| 种子 | `eval_info.json` 里逐回合 seed = 1000…1499（训练 seed 1000） | seed = 1000…1499（训练 `--seed=100000`） |
| 环境 | `AlohaTransferCube-v0`（gym-aloha），`episode_length=400`，`fps=50`，`obs_type=pixels_agent_pos`，只用 `top` 相机（480×640） | `PushT-v0`（gym-pusht），`episode_length=300`，`fps=10`，`obs_type=pixels_agent_pos`，策略输入图像 96×96 |
| 环境包版本 | **未说明**（只说训练用 LeRobot@3c0a209；gym-aloha/gym-pusht 版本未写） | **未说明** |
| 训练 | 80k 步检查点（A100 约 1h45）；数据集 `lerobot/aloha_sim_transfer_cube_human` | 175k 步检查点；数据集 `lerobot/pusht` |
| 评估硬件/渲染器 | **未说明**（`eval_info.json` 里 `eval_ep_s`≈5.15 s/回合，批 50） | **未说明**（`eval_ep_s`≈1.46 s/回合） |
| chunk_size / horizon | `chunk_size=100` | `horizon=16` |
| **n_action_steps（执行长度 k 的默认值）** | **100**（=整个块，开环执行 100 步再重新观察；400 步回合里只重新规划 4 次） | **8**（`n_obs_steps=2`，每 8 步重新预测） |
| 时间集成（temporal ensembling） | **未开**（`temporal_ensemble_coeff=null`；模型卡没提到时间集成） | 不适用（DP 无此选项） |
| 其他关键 config | `use_vae=true`，`latent_dim=32`，`kl_weight=10`，`vision_backbone=resnet18`，`dim_model=512`，`n_encoder_layers=4`，`use_amp=false` | `DDPM`，`num_train_timesteps=100`，`num_inference_steps=null`（=推理也用 100 步去噪），`prediction_type=epsilon`，`vision_backbone=resnet18`，`crop_shape=[84,84]`（评估时中心裁剪），`down_dims=[512,1024,2048]`，参数量 262.7M |
| “官方默认推理设置” | 即上表 config：k=100，不做时间集成 | 即上表 config：k=8 |

要点：
1. 官方数字 **用的就是各模型 config 里的默认 k**（ACT=100、DP=8），不是另外调过的值。
2. 官方评估的**种子 1000–1499 落在本项目的调试/调参区间（0–9999）内**，所以“官方数字”不能当作我们最终评估的参照种子；最终评估用 ≥100000（见 CLAUDE.md §5）。
3. 原 ACT 论文推荐的时间集成（系数 0.01）在这个 hub 模型的默认设置里是**关闭**的；若后续要比较“k=1 + 时间集成”，需要显式打开 `temporal_ensemble_coeff`（此时 LeRobot 要求 `n_action_steps=1`）。
4. 模型卡指出：重构后的代码可能有 bug/与原仓库评估启发式不同，所以 83% 与原论文数字不可直接比较。

## 6. 冒烟测试与实测每回合耗时

`scripts/smoke.sh`（`results/smoke/`；官方默认推理设置，GPU，种子 0、1；每回合 <3 分钟，所以每个模型都跑了 2 回合）。

**最终版本（GPU 渲染，`results/smoke/smoke_episodes.csv`）**

| 模型 | seed | 成功 | 步数 | 回合墙钟 | 策略推理 | 环境步进+渲染 |
|---|---|---|---|---|---|---|
| ACT | 0 | 否 | 400/400 | 6.0 s | 0.5 s | 4.9 s |
| ACT | 1 | 是 | 264/400 | 3.8 s | 0.2 s | 3.2 s |
| DP | 0 | 是 | 234/300 | 47.4 s | 46.9 s | 0.4 s |
| DP | 1 | 否 | 300/300 | 59.1 s | 58.4 s | 0.5 s |

**对照（CPU llvmpipe 渲染，`results/smoke_llvmpipe/`）**：ACT 回合 18.7 s / 18.5 s（环境 17.5 s / 17.9 s），两回合都成功（257、262 步）；DP 41.5 s / 60.5 s（成功 / 失败，202 / 300 步）。

2 个回合不是成功率估计，只说明“能跑、不报错”。

耗时规划要点：
- **ACT**：每回合 ≈4–6 s（环境渲染占 ~80%）；默认 k=100 时整回合策略推理只有 0.2–0.8 s（含 3–4 次块预测，首回合含预热）；k 变小会让预测次数增加（k=1 时每步一次），推理占比会上升，具体多少**未实测**，留给后续任务。
- **DP**：策略推理占 ~99%。一次动作块预测（100 步 DDPM 去噪，262.7M 参数）实测 **约 1.56 s**（`scripts/bench_render.py dp_infer`，稳态 1.56–1.57 s）。默认 k=8 时 300 步回合约 38 次预测 ≈ 47–60 s。**若 k=1，则每步一次预测，回合约 300×1.56 s ≈ 8 分钟**；k=2 ≈ 4 分钟/回合。DP 的 k 扫描会非常贵——后续任务排计划时必须考虑（例如减少 DP 的 k 取值、降低回合数，或评估是否允许减少去噪步数——但那就偏离了“官方默认推理设置”，需统筹决定）。
- 模型加载：ACT ≈15 s、DP ≈33 s（含首次下载与 CUDA 初始化；ACT 首次还会下载 ResNet18 权重）。
- 首帧图像检查（`results/smoke/first_obs_*.png`）：ACT 480×640×3，均值 39.9、标准差 50.8、197 个不同取值——黑背景上有灰色桌面、两只机械臂和红色方块，画面正常；DP 96×96×3，均值 249.3、标准差 21.7、89 个不同取值——白底上有灰色 T 块、绿色目标和蓝色智能体点，画面正常。标准差都明显大于 0，不是全黑/单色。

## 7. 已知小问题

- 控制台会有 `huggingface_hub` “unauthenticated requests” 提示（未设 HF_TOKEN，不影响下载）和 gymnasium 的 `env.task_description … deprecated` 警告（0.3.3 与 gymnasium 0.29 的已知噪声）。
- `uv pip install` 时会提示 `huggingface-hub==1.33.0 does not have an extra named 'cli'/'hf-transfer'`（lerobot 0.3.3 声明了这两个 extra，新版 hub 已取消），无影响。
- WSL 发行版需要保持运行（见记忆里的 keep-alive 说明）；`wsl.exe` 的输出带 `\0`，用 `| tr -d '\0'`。
