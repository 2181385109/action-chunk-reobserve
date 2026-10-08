"""T10b 自训练 ACT 正式评估辅助：生成配置 / 列出格子 / 判断是否完成 / 估算耗时 / 登记检查 / 检查点哈希。run_main_own.sh 调用。

  python scripts/main_own_tools.py gen  [--out configs/main_own]   # 9 格 + 基础水平测试的配置
  python scripts/main_own_tools.py list [--select 1-5]             # 9 个格子名，一行一个（--select：范围或逗号分隔的格子名）
  python scripts/main_own_tools.py done <目录> <回合数>               # 已完成 exit 0，否则 exit 1
  python scripts/main_own_tools.py eta  [--select 1-5|none] [--no-baseline]   # 预计耗时（含基础水平测试，除非 --no-baseline；--select none = 只估基础水平测试）
  python scripts/main_own_tools.py seeds | ckpt                    # 打印正式种子范围 / 登记的检查点路径
  python scripts/main_own_tools.py guard                           # PREREGISTRATION_OWN.md 已提交？（没有 exit 3）
  python scripts/main_own_tools.py verify-ckpt                     # 检查点 model.safetensors 的 SHA-256 与登记一致？（不一致 exit 4）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chunkre import main_own as mo  # noqa: E402
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.pilot import cell_done  # noqa: E402


def _wall_main(root: Path = ROOT) -> dict[str, float]:
    """预训练模型正式实验（results/main/）里这 9 格各自的总墙钟（秒，200 回合）；缺的不在字典里。"""
    out = {}
    for c in mo.cells():
        p = root / "results" / "main" / c.name / "summary.json"
        if p.exists():
            out[c.name] = float(json.loads(p.read_text(encoding="utf-8"))["metrics"]["total_batch_wall_s"])
    return out


def _wall_repro(root: Path = ROOT) -> float | None:
    p = root / "results" / "repro" / "act" / "summary.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["metrics"]["total_batch_wall_s"])


def eta_measured(cell_list, with_baseline: bool, root: Path = ROOT, load_s: float = 15.0) -> float | None:
    """按预训练模型实测的用时估算（自训练模型失败回合多时回合更长，实际可能略高）；缺数据返回 None。"""
    w = _wall_main(root)
    if any(c.name not in w for c in cell_list):
        return None
    total = sum(w[c.name] + load_s for c in cell_list)
    if with_baseline:
        r = _wall_repro(root)
        if r is None:
            return None
        total += r + load_s
    return total


def eta_table(cell_list, with_baseline: bool) -> float:
    from pilot_tools import eta_seconds   # T2 耗时表（每回合按跑满步数计，偏保守）+ 每格开销
    t = eta_seconds(cell_list, mo.OWN_N, mo.OWN_BATCH)
    if with_baseline:
        base = mo.Cell(mo.BASELINE_NAME, "k100", "fixed", mo.BASELINE_K, "none", "none", 0.0)
        t += eta_seconds([base], mo.BASELINE_N, mo.OWN_BATCH)
    return t


def checkpoint_sha256(path: str = mo.CHECKPOINT) -> str | None:
    f = Path(path) / "model.safetensors"
    if not f.is_file():
        return None
    from chunkre.runner import file_sha256

    return file_sha256(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "list", "done", "eta", "seeds", "ckpt", "guard", "verify-ckpt"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out", default="configs/main_own")
    ap.add_argument("--select", default=None, help="格子子集：范围 1-5（从 1 数）或逗号分隔的格子名；默认全部 9 格")
    ap.add_argument("--no-baseline", action="store_true", help="eta：不含基础水平测试")
    a = ap.parse_args(argv)
    names = [c.name for c in mo.cells()]

    if a.cmd == "gen":
        out = Path(a.out)
        paths = mo.write_configs(out if out.is_absolute() else ROOT / out)
        print(f"写出 {len(paths)} 个配置 -> {a.out}（9 格种子 {mo.OWN_SEEDS}；基础水平测试种子 {mo.BASELINE_SEEDS}；"
              f"检查点 {mo.CHECKPOINT}）")
    elif a.cmd == "list":
        print("\n".join(mo.parse_selection(a.select, names)))
    elif a.cmd == "seeds":
        print(mo.OWN_SEEDS)
    elif a.cmd == "ckpt":
        print(mo.CHECKPOINT)
    elif a.cmd == "done":
        sys.exit(0 if cell_done(Path(a.args[0]), int(a.args[1])) else 1)
    elif a.cmd == "guard":
        from chunkre.seed_guard import check_seeds, preregistration_committed

        if not preregistration_committed(model="act_own"):
            print("PREREGISTRATION_OWN.md 尚未提交：登记提交之前不能对自训练模型做任何评估（含基础水平测试）。", file=sys.stderr)
            sys.exit(3)
        check_seeds(parse_seeds(mo.OWN_SEEDS), model="act_own")
        check_seeds(parse_seeds(mo.BASELINE_SEEDS), model="act_own")
        print("PREREGISTRATION_OWN.md 已提交，种子检查通过")
    elif a.cmd == "verify-ckpt":
        got = checkpoint_sha256()
        if got is None:
            print(f"检查点不存在：{mo.CHECKPOINT}/model.safetensors", file=sys.stderr)
            sys.exit(4)
        if got != mo.CHECKPOINT_SHA256:
            print(f"检查点哈希与登记不一致：\n  实际 {got}\n  登记 {mo.CHECKPOINT_SHA256}", file=sys.stderr)
            sys.exit(4)
        print(f"检查点哈希与登记一致：{got}")
    else:
        sel = set() if a.select == "none" else set(mo.parse_selection(a.select, names))   # none = 只估基础水平测试
        cl = [c for c in mo.cells() if c.name in sel]
        wb = not a.no_baseline
        what = (f"{len(cl)} 格 × {mo.OWN_N} 回合" if cl else "") + (" + " if cl and wb else "") +             (f"基础水平测试 {mo.BASELINE_N} 回合" if wb else "")
        m = eta_measured(cl, wb)
        if m is not None:
            print(f"{what}：按预训练模型同样格子的实测用时估算约 {m / 3600:.1f} 小时（{m / 60:.0f} 分钟；PC 上别同时跑其他 GPU 任务）")
        try:
            t = eta_table(cl, wb)
            print(f"{what}：按 T2 耗时表保守估算约 {t / 3600:.1f} 小时（每回合按跑满 400 步计）")
        except Exception as e:   # 没有耗时表时只给实测估算
            if m is None:
                raise
            print(f"    （T2 耗时表估算失败：{e}）")


if __name__ == "__main__":
    main()
