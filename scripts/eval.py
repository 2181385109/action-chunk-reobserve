"""统一评估命令行入口（T2）。后续所有实验都用它。

用法（WSL 内，先 source scripts/env.sh；或用 scripts/eval.sh 包装）：
  python scripts/eval.py --config configs/act_default.yaml --out results/<实验名>
  python scripts/eval.py --config configs/act_default.yaml --out results/x --k 10 --seeds 0-9 --batch-size 10
  python scripts/eval.py --config configs/act_te.yaml --out results/x

命令行参数覆盖 YAML 里的同名字段。输出（CLAUDE.md §6）：
  <out>/config.json   配置 + 策略实际参数 + 实际 GL_RENDERER + 版本
  <out>/episodes.csv  一回合一行
  <out>/summary.json  成功率 + Wilson 95% 区间、平均调用次数、平均平滑度（可由 scripts/summarize.py 重算）
同一输出目录再次运行 = 断点续跑（已写入 CSV 的批次跳过）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="configs/*.yaml")
    ap.add_argument("--out", required=True, help="输出目录（相对仓库根或绝对路径）")
    ap.add_argument("--model", choices=["act", "dp"])
    ap.add_argument("--k", type=int, help="执行长度 k（不传 = 模型默认：ACT 100 / DP 8）")
    ap.add_argument("--mode", choices=["fixed", "te"])
    ap.add_argument("--seeds", help="例如 0-9 或 1000-1499 或 0,1,2")
    ap.add_argument("--batch-size", type=int, dest="batch_size", help="同时运行的环境数 B")
    ap.add_argument("--max-steps", type=int, dest="max_steps", help="只用于检查：每回合最多步数")
    ap.add_argument("--name")
    ap.add_argument("--device")
    ap.add_argument("--allow-cpu-render", action="store_true", default=None, dest="allow_cpu_render")
    ap.add_argument("--no-deterministic", action="store_false", default=None, dest="deterministic")
    ap.add_argument("--log-traj", action="store_true", default=None, dest="log_traj",
                    help="逐步轨迹 -> <out>/traj/batch_*.npz（不影响结果）")
    ap.add_argument("--save-frames", dest="save_frames", help="这些种子导出画面 PNG + GIF -> <out>/frames/，如 0 或 0,3")
    ap.add_argument("--frame-every", type=int, dest="frame_every", help="每隔几步存一张 PNG（默认 10）")
    a = ap.parse_args(argv)
    overrides = {k: getattr(a, k) for k in ["model", "k", "mode", "seeds", "batch_size", "max_steps", "name",
                                             "device", "allow_cpu_render", "deterministic", "log_traj",
                                             "save_frames", "frame_every"]}
    cfg = load_config(a.config, overrides)
    if not cfg.name:
        cfg.name = Path(a.out).name
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out

    from chunkre.runner import run_eval  # 延迟导入：只有真正跑时才加载 torch/lerobot

    summary = run_eval(cfg, out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
