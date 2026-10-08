"""T11a 自适应阈值试跑辅助：生成配置 / 列出格子 / 判断是否完成 / 估算耗时。run_tune_adaptive.sh 调用。

  python scripts/tune_adaptive_tools.py gen  [--out configs/tune_adaptive] [--seeds 6000-6049] [--batch 10] [--max-steps N]
  python scripts/tune_adaptive_tools.py list [--select 1-11]          # 22 个格子名，一行一个
  python scripts/tune_adaptive_tools.py done <目录> <回合数>            # 已完成 exit 0，否则 exit 1
  python scripts/tune_adaptive_tools.py eta  [--select ...] [--seeds ...]   # 预计耗时
  python scripts/tune_adaptive_tools.py n    [--seeds ...]              # 种子个数
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre import tune_adaptive as ta  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.main_own import parse_selection  # noqa: E402
from chunkre.pilot import cell_done  # noqa: E402

LOAD_S = 15.0     # 每格启动 + 加载模型（秒）
SAFETY = 1.1      # adaptive 格子：阈值小的组合换计划多、失败回合可能更长，留余量


def _per_episode(path: Path) -> float | None:
    p = path / "summary.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))["metrics"]
    return float(m["total_batch_wall_s"]) / int(m["n_episodes"])


def per_episode_s(c: ta.TuneCell, root: Path = ROOT) -> float:
    """每回合墙钟（B=10 时按批平摊）：参照格用 T6a 试跑同一格的实测；adaptive 格两种条件都用 T11a A3 推 0.06（m=10）的实测 × SAFETY
    （A3 无扰动那格和 CPU 渲染检查并行跑、用时偏高，不用；推 0.06 失败多、回合长，用于无扰动偏保守）。"""
    if c.adaptive:
        v = _per_episode(root / "results" / "t11_checks" / "a3_push0.06")
        return v * SAFETY if v is not None else 8.0
    v = _per_episode(root / "results" / "pilot" / c.name)
    return v if v is not None else 8.0


def eta_seconds(cell_list, n: int) -> float:
    return sum(per_episode_s(c) * n + LOAD_S for c in cell_list)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "list", "done", "eta", "n"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default="configs/tune_adaptive")
    ap.add_argument("--seeds", default=ta.TUNE_SEEDS)
    ap.add_argument("--batch", type=int, default=ta.TUNE_BATCH)
    ap.add_argument("--max-steps", type=int, default=None, help="只用于脚本自检（极小配置）")
    ap.add_argument("--select", default=None, help="格子子集：范围 1-11（从 1 数）或逗号分隔的格子名；默认全部 22 格")
    a = ap.parse_args(argv)
    names = [c.name for c in ta.cells()]

    if a.cmd == "gen":
        out = Path(a.out)
        out = out if out.is_absolute() else ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        for c in ta.cells():
            (out / f"{c.name}.yaml").write_text(ta.config_yaml(c, a.seeds, a.batch, a.max_steps),
                                                encoding="utf-8", newline="\n")
        print(f"写出 {len(names)} 个配置 -> {a.out}（种子 {a.seeds}，B={a.batch}，tau 候选 {list(ta.TAU_CANDIDATES)}）")
    elif a.cmd == "list":
        print("\n".join(parse_selection(a.select, names)))
    elif a.cmd == "done":
        sys.exit(0 if cell_done(Path(a.args[0]), int(a.args[1])) else 1)
    elif a.cmd == "n":
        print(len(parse_seeds(a.seeds)))
    else:
        sel = set(parse_selection(a.select, names))
        cl = [c for c in ta.cells() if c.name in sel]
        n = len(parse_seeds(a.seeds))
        t = eta_seconds(cl, n)
        print(f"{len(cl)} 格 × {n} 回合：按实测用时估算约 {t / 3600:.1f} 小时（{t / 60:.0f} 分钟；"
              f"参照格按 T6a 试跑、自适应格按 T11a A3 推 0.06 实测 ×{SAFETY}）")


if __name__ == "__main__":
    main()
