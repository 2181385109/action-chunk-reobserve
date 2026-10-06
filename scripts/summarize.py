"""由 episodes.csv 重新计算汇总，并与 summary.json 比对（CLAUDE.md §6：汇总必须能从逐回合 CSV 重算）。

用法：
  python scripts/summarize.py results/<实验名>            # 打印重算结果，并与 summary.json 比对
  python scripts/summarize.py results/<实验名> --write    # 用重算结果覆盖 summary.json 的 metrics
不需要 GPU / 仿真，Windows 或 WSL 里有 numpy + pyyaml 即可。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.metrics import summarize_csv  # noqa: E402


def same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def check(run_dir: Path, write: bool = False) -> bool:
    run_dir = Path(run_dir)
    m = summarize_csv(run_dir / "episodes.csv")
    sp = run_dir / "summary.json"
    ok = True
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        diff = {k: (s["metrics"].get(k), v) for k, v in m.items() if not same(s["metrics"].get(k), v)}
        ok = not diff
        print(f"[summarize] {run_dir}: {'一致' if ok else '不一致'}" + ("" if ok else f" {diff}"))
        if write:
            s["metrics"] = m
            sp.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"[summarize] {run_dir}: 没有 summary.json")
    print(json.dumps(m, indent=2, ensure_ascii=False))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ok = all([check(Path(d), a.write) for d in a.run_dirs])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
