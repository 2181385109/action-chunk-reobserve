"""T13 run_failure.sh 的辅助：格子列表、每格 eval.py 参数、是否完成、耗时估算。

  python scripts/failure_tools.py list [--cells a,b]          # 格子名（空格分隔）
  python scripts/failure_tools.py args <格子> --seeds 7000-7049  # 该格 eval.py 额外参数（k、扰动、存画面的种子）
  python scripts/failure_tools.py done <目录> <回合数>           # exit 0 = 已完成（episodes.csv 行数 = 回合数且有 summary.json）
  python scripts/failure_tools.py eta <输出根> --seeds 7000-7049 [--cells ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.config import parse_seeds  # noqa: E402

CELLS = ["k20__none", "k100__none", "k20__push0.06", "k100__push0.06", "k20__delay10", "k50__delay10", "k100__delay10"]
PUSH_ARGS = ["--perturb", "push", "--perturb-strength", "0.06", "--perturb-param", "t_lo=20", "--perturb-param", "t_hi=100"]
DELAY_ARGS = ["--perturb", "action_delay", "--perturb-strength", "10"]
FRAME_SEEDS = 3          # 每格前 3 个种子存画面
FRAME_SEEDS_PUSH = 10    # 推的两格前 10 个种子（为了有足够的 k20 成功 / k100 失败配对可做拼图，见 definitions.md §6）
FRAME_EVERY = 3
SEC_PER_EP = 6.5         # 实测：调试种子 0–9、B=10、两个进程并行时每批约 73–76 s；单进程估 6.5 s/回合（含轨迹 + 存图）


def cell_args(name: str, seeds: list[int]) -> list[str]:
    k = name.split("__")[0][1:]
    cond = name.split("__")[1]
    a = ["--k", k]
    if cond.startswith("push"):
        a += PUSH_ARGS
    elif cond.startswith("delay"):
        a += DELAY_ARGS
    nf = FRAME_SEEDS_PUSH if cond.startswith("push") else FRAME_SEEDS
    a += ["--save-frames", ",".join(str(s) for s in seeds[:nf]), "--frame-every", str(FRAME_EVERY)]
    return a


def n_done(d: Path) -> int:
    p = d / "episodes.csv"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "args", "done", "eta"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--seeds", default="7000-7049")
    ap.add_argument("--cells", default="")
    a = ap.parse_args(argv)
    cells = [c for c in a.cells.split(",") if c] or CELLS
    unknown = set(cells) - set(CELLS)
    if unknown:
        raise SystemExit(f"未知格子 {sorted(unknown)}；可用 {CELLS}")
    seeds = parse_seeds(a.seeds)
    if a.cmd == "list":
        print(" ".join(cells))
    elif a.cmd == "args":
        print(" ".join(cell_args(a.rest[0], seeds)))
    elif a.cmd == "done":
        d = Path(a.rest[0])
        sys.exit(0 if n_done(d) >= int(a.rest[1]) and (d / "summary.json").exists() else 1)
    else:
        out = Path(a.rest[0])
        left = sum(max(0, len(seeds) - n_done(out / c)) for c in cells)
        print(f"    还要跑 {left} 回合（{len(cells)} 格 × {len(seeds)}），按 {SEC_PER_EP} s/回合估约 {left * SEC_PER_EP / 60:.0f} 分钟")


if __name__ == "__main__":
    main()
