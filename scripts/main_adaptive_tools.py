"""T11b 自适应规则正式评估辅助：生成配置 / 列出格子 / 判断是否完成 / 估算耗时 / 登记检查 / 参数核对。run_main_adaptive.sh 调用。

  python scripts/main_adaptive_tools.py gen [--out configs/main_adaptive] [--m M --tau T --h-cmp H --min-gap G]
                                                              # 2 格配置；不写参数 = 登记的参数（参数覆盖只用于演示“参数不一致被拦”）
  python scripts/main_adaptive_tools.py list                  # 2 个格子名，一行一个
  python scripts/main_adaptive_tools.py done <目录> <回合数>     # 已完成 exit 0，否则 exit 1
  python scripts/main_adaptive_tools.py eta [--select ad__none,...]   # 预计耗时
  python scripts/main_adaptive_tools.py seeds | params         # 打印正式种子范围 / 登记的自适应参数
  python scripts/main_adaptive_tools.py guard                  # PREREGISTRATION_ADAPTIVE.md 已提交？（没有 exit 3）
  python scripts/main_adaptive_tools.py verify-params [--out configs/main_adaptive]
                                                              # 目录里每个配置的 m、tau、h_cmp、min_gap 与登记文件里的数字逐项相同？（不同 exit 4）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chunkre import main_adaptive as ma  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.pilot import cell_done  # noqa: E402


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def eta_seconds(names: list[str], load_s: float = 15.0) -> tuple[float | None, float | None]:
    """(按试跑实测, 按 results/main 的 k100 实测) 估算秒数；缺数据 None。

    试跑实测：results/tune_adaptive 里 ad_m10_q90__{none,push0.06} 各自 50 回合的总墙钟 × (200/50)。
    参照：results/main/k100__* 的 200 回合墙钟（自适应每步多一次策略调用，实际会更长）。"""
    tune = ROOT / "results" / "tune_adaptive"
    t_tune = t_main = None
    try:
        tot = 0.0
        for n in names:
            tag = n.split("__", 1)[1]
            s = json.loads((tune / f"ad_m10_q90__{tag}" / "summary.json").read_text(encoding="utf-8"))
            tot += float(s["metrics"]["total_batch_wall_s"]) * (ma.ADAPTIVE_N / 50) + load_s
        t_tune = tot
    except Exception:   # noqa: BLE001
        pass
    try:
        tot = 0.0
        for n in names:
            tag = n.split("__", 1)[1]
            s = json.loads((ROOT / ma.CONTROL_DIR / f"{ma.CONTROL_POLICY}__{tag}" / "summary.json").read_text(encoding="utf-8"))
            tot += float(s["metrics"]["total_batch_wall_s"]) + load_s
        t_main = tot
    except Exception:   # noqa: BLE001
        pass
    return t_tune, t_main


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "list", "done", "eta", "seeds", "params", "guard", "verify-params"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default="configs/main_adaptive")
    ap.add_argument("--select", default=None, help="eta：逗号分隔的格子名；默认全部")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--h-cmp", type=int, default=None, dest="h_cmp")
    ap.add_argument("--min-gap", type=int, default=None, dest="min_gap")
    a = ap.parse_args(argv)
    names = ma.cell_names()

    if a.cmd == "gen":
        params = dict(ma.ADAPTIVE_PARAMS)
        for k in ma.PARAM_KEYS:
            if getattr(a, k) is not None:
                params[k] = getattr(a, k)
        paths = ma.write_configs(_abs(a.out), params)
        print(f"写出 {len(paths)} 个配置 -> {a.out}（种子 {ma.ADAPTIVE_SEEDS}；自适应参数 {params}"
              + ("" if params == ma.ADAPTIVE_PARAMS else "；**与登记常量不同**") + "）")
    elif a.cmd == "list":
        print("\n".join(names))
    elif a.cmd == "seeds":
        print(ma.ADAPTIVE_SEEDS)
    elif a.cmd == "params":
        print(json.dumps(ma.registered_params(ROOT)))
    elif a.cmd == "done":
        sys.exit(0 if cell_done(Path(a.args[0]), int(a.args[1])) else 1)
    elif a.cmd == "guard":
        from chunkre.seed_guard import check_seeds, preregistration_committed

        if not preregistration_committed(model="act_adaptive"):
            print("PREREGISTRATION_ADAPTIVE.md 尚未提交：登记提交之前不能用最终种子评估自适应规则。", file=sys.stderr)
            sys.exit(3)
        check_seeds(parse_seeds(ma.ADAPTIVE_SEEDS), model="act_adaptive")
        print("PREREGISTRATION_ADAPTIVE.md 已提交，种子检查通过")
    elif a.cmd == "verify-params":
        reg = ma.registered_params(ROOT)
        d = _abs(a.out)
        paths = [d / f"{n}.yaml" for n in names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"缺少配置：{missing}（先运行 gen）", file=sys.stderr)
            sys.exit(4)
        bad = ma.check_run_params(paths, reg)
        if bad:
            print("运行参数与 PREREGISTRATION_ADAPTIVE.md 不一致：", file=sys.stderr)
            for b in bad:
                print("  " + b, file=sys.stderr)
            sys.exit(4)
        print(f"运行参数与登记逐项相同：m={reg['m']}, tau={reg['tau']}, h_cmp={reg['h_cmp']}, min_gap={reg['min_gap']}"
              f"（{len(paths)} 个配置）")
    else:
        sel = names if not a.select else a.select.split(",")
        bad = [s for s in sel if s not in names]
        if bad:
            sys.exit(f"未知格子名：{bad}")
        t_tune, t_main = eta_seconds(sel)
        what = f"{len(sel)} 格 × {ma.ADAPTIVE_N} 回合"
        if t_tune is not None:
            print(f"{what}：按阈值试跑里同参数格子的实测用时估算约 {t_tune / 3600:.1f} 小时（{t_tune / 60:.0f} 分钟；PC 上别同时跑其他 GPU 任务）")
        if t_main is not None:
            print(f"{what}：作为下限参考，k=100 同样格子的实测用时约 {t_main / 3600:.1f} 小时（自适应每步多算几次策略，实际更长）")
        if t_tune is None and t_main is None:
            print(f"{what}：没有可参考的实测用时")


if __name__ == "__main__":
    main()
