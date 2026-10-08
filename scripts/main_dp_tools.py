"""T9 DP 正式评估辅助：生成配置 / 列出格子 / 判断是否完成 / 估算耗时。run_main_dp.sh 调用。

  python scripts/main_dp_tools.py gen  [--out configs/main_dp]
  python scripts/main_dp_tools.py list [--select 1-6]      # 格子名，一行一个（--select：范围或逗号分隔的格子名）
  python scripts/main_dp_tools.py done <格子目录> <回合数>     # 已完成 exit 0，否则 exit 1
  python scripts/main_dp_tools.py eta  [--select 1-6]      # 预计耗时（两种估算）
  python scripts/main_dp_tools.py seeds                    # 打印正式种子范围
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chunkre import main_dp as md  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.pilot import cell_done  # noqa: E402


def pilot_batch_seconds(root: Path = ROOT) -> dict[str, float]:
    """DP 试跑（results/pilot_dp/）里每格实测的每批墙钟（秒，试跑每格 1 批）；没有试跑数据的格子不在字典里。"""
    out = {}
    for c in md.cells():
        p = root / "results" / "pilot_dp" / c.name / "episodes.csv"
        if not p.exists():
            continue
        with open(p, newline="", encoding="utf-8") as f:
            bw = {r["batch_idx"]: float(r["batch_wall_s"]) for r in csv.DictReader(f)}
        if bw:
            out[c.name] = sum(bw.values()) / len(bw)
    return out


def eta_pilot_based(cell_list, n: int, B: int, load_s: float = 15.0) -> float | None:
    """试跑实测的每批用时 × 批数 + 每格加载开销；缺任何一格的试跑数据就返回 None。"""
    bs = pilot_batch_seconds()
    if any(c.name not in bs for c in cell_list):
        return None
    nb = -(-n // B)
    return sum(bs[c.name] * nb + load_s for c in cell_list)


def eta_table_based(cell_list, n: int, B: int) -> float:
    from pilot_tools import eta_seconds   # T2 耗时表（每回合按跑满步数计，偏保守）+ 每格开销
    return eta_seconds(cell_list, n, B, model="dp")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "list", "done", "eta", "seeds"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default="configs/main_dp")
    ap.add_argument("--select", default=None, help="格子子集：范围 1-6（从 1 数）或逗号分隔的格子名；默认全部 12 格")
    a = ap.parse_args(argv)
    names = [c.name for c in md.cells()]

    if a.cmd == "gen":
        out = Path(a.out)
        paths = md.write_configs(out if out.is_absolute() else ROOT / out)
        print(f"写出 {len(paths)} 个配置 -> {a.out}（种子 {md.MAIN_DP_SEEDS}，B={md.MAIN_DP_BATCH}）")
    elif a.cmd == "list":
        print("\n".join(md.parse_selection(a.select, names)))
    elif a.cmd == "seeds":
        print(md.MAIN_DP_SEEDS)
    elif a.cmd == "done":
        sys.exit(0 if cell_done(Path(a.args[0]), int(a.args[1])) else 1)
    else:
        sel = set(md.parse_selection(a.select, names))
        cl = [c for c in md.cells() if c.name in sel]
        n, B = len(parse_seeds(md.MAIN_DP_SEEDS)), md.MAIN_DP_BATCH
        p = eta_pilot_based(cl, n, B)
        if p is not None:
            print(f"{len(cl)} 格 × {n} 回合：按 DP 试跑实测每批用时估算约 {p / 3600:.1f} 小时"
                  "（试跑时的 GPU 负载下；PC 上别同时跑其他 GPU 任务）")
        try:
            t = eta_table_based(cl, n, B)
            print(f"{len(cl)} 格 × {n} 回合：按 T2 耗时表保守估算约 {t / 3600:.1f} 小时")
        except Exception as e:   # 没有耗时表时只给试跑估算
            if p is None:
                raise
            print(f"    （T2 耗时表估算失败：{e}）")


if __name__ == "__main__":
    main()
