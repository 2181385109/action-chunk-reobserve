"""试跑辅助：生成配置 / 判断格子是否完成 / 估算耗时。run_pilot.sh（T6a，ACT）和 run_pilot_dp.sh（T8b，DP）调用。

  python scripts/pilot_tools.py [--grid act|dp] gen  [--out configs/pilot] [--seeds 5000-5019] [--batch-size 10]
  python scripts/pilot_tools.py [--grid act|dp] list                      # 格子名，一行一个
  python scripts/pilot_tools.py done <格子目录> <回合数>                   # 已完成 exit 0，否则 exit 1
  python scripts/pilot_tools.py [--grid act|dp] eta  [--n 20] [--B 10] [--cells a,b,...]
--grid 默认 act（T6a ACT 试跑，59 格）；dp = T8b DP 试跑（40 格，默认种子 5000–5049、B=50、输出 configs/pilot_dp）。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chunkre import pilot, pilot_dp  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.pilot import cell_done  # noqa: E402

# 每个格子的固定开销（加载模型 + 建环境，秒）和观测噪声每批额外开销（秒）。只用于开头打印预计耗时。
# ACT：LOAD_S 来自 T6a 极小试跑实测（每格总用时比批用时多约 12 s）；噪声额外开销取自 T5 N4（每批 10 张 480×640 图 × 400 步）。
# DP：PushT 图像只有 96×96，噪声开销按每步约 10 ms 估（50 个环境），每批 300 步约 3 s，取 5 s。
LOAD_S = 15.0
NOISE_EXTRA_PER_BATCH_S = {"act": 40.0, "dp": 5.0}
GRIDS = {
    "act": dict(mod=pilot, seeds=pilot.PILOT_SEEDS, B=pilot.PILOT_BATCH, out="configs/pilot", model="act"),
    "dp": dict(mod=pilot_dp, seeds=pilot_dp.PILOT_DP_SEEDS, B=pilot_dp.PILOT_DP_BATCH, out="configs/pilot_dp",
               model="dp"),
}


def eta_seconds(cell_list, n: int, B: int, model: str = "act") -> float:
    import bench_timing as bt   # 只用它的纯计算函数（不加载 torch）

    raw = bt.OUT / f"timing_raw_{model}.csv"
    with open(raw, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if int(r["B"]) == B]
    if not rows:   # 没有这个 B 的实测：取最接近的 B 的单步耗时近似（只在试跑 / 检查用了非常规 B 时发生）
        with open(raw, newline="", encoding="utf-8") as f:
            allrows = list(csv.DictReader(f))
        rows = [min(allrows, key=lambda r: abs(int(r["B"]) - B))]
        print(f"（没有 B={B} 的实测耗时，用 B={rows[0]['B']} 的近似）")
    r = {key: (float(v) if key != "model" else v) for key, v in rows[0].items()}
    r["B"] = B
    total = 0.0
    for c in cell_list:
        k = "te" if c.mode == "te" else c.k
        total += bt.est_episode_s(model, k, r) * n + LOAD_S
        if c.kind == "obs_noise":
            total += math.ceil(n / B) * NOISE_EXTRA_PER_BATCH_S[model]
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "list", "done", "eta"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--grid", choices=list(GRIDS), default="act")
    ap.add_argument("--out", default=None, help="gen：配置输出目录（默认 act: configs/pilot，dp: configs/pilot_dp）")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    ap.add_argument("--n", type=int, default=None, help="eta：每格回合数（默认 = 种子数）")
    ap.add_argument("--B", type=int, default=None, help="eta：批大小（默认 = batch-size）")
    ap.add_argument("--cells", default=None, help="eta：只算这些格子（逗号分隔）")
    a = ap.parse_args(argv)
    g = GRIDS[a.grid]
    mod = g["mod"]
    seeds = a.seeds or g["seeds"]
    bsz = a.batch_size or g["B"]

    if a.cmd == "gen":
        out = Path(a.out or g["out"])
        paths = mod.write_configs(out if out.is_absolute() else ROOT / out, seeds, bsz)
        print(f"写出 {len(paths)} 个配置 -> {out}（种子 {seeds}，B={bsz}）")
    elif a.cmd == "list":
        print("\n".join(c.name for c in mod.cells()))
    elif a.cmd == "done":
        sys.exit(0 if cell_done(Path(a.args[0]), int(a.args[1])) else 1)
    else:
        want = set(a.cells.split(",")) if a.cells else None
        cl = [c for c in mod.cells() if want is None or c.name in want]
        n = a.n or len(parse_seeds(seeds))
        B = a.B or bsz
        s = eta_seconds(cl, n, B, g["model"])
        print(f"{len(cl)} 格 × {n} 回合：预计约 {s / 3600:.1f} 小时（{s / 60:.0f} 分钟）"
              f"——按 T2 耗时表估算（每批按跑满全部步数计，偏保守）+ 每格 {LOAD_S:.0f} s 开销 + 噪声格每批 "
              f"{NOISE_EXTRA_PER_BATCH_S[g['model']]:.0f} s")


if __name__ == "__main__":
    main()
