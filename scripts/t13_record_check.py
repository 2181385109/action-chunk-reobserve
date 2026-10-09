"""T13：证明“加了指尖记录前后，动作逐位相同”（CPU 渲染，种子 0–4）。

由 scripts/t13_record_check.sh 调用：同一配置分别用加记录之前的代码（git archive 出的旧提交）和当前代码跑
eval.py --log-traj，然后比较：
  - episodes.csv：除墙钟列（TIME_COLUMNS）外逐列字符串相同；
  - traj/*.npz：旧代码有的每个数组（action、agent_pos、cube_pos、chunk_sel、policy_action …）逐字节相同；
  - 新代码多出 finger_tip（形状 (T, B, 4, 3)，存活步没有 nan）。
  python scripts/t13_record_check.py compare <旧输出根> <新输出根> <条件名...>  -> results/failure_checks/record_check.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkre.metrics import TIME_COLUMNS, read_episodes  # noqa: E402

OUT = ROOT / "results" / "failure_checks"


def compare_cell(old: Path, new: Path) -> dict:
    ro, rn = read_episodes(old / "episodes.csv"), read_episodes(new / "episodes.csv")
    cols = [c for c in ro[0] if c not in TIME_COLUMNS] if ro else []
    csv_same = len(ro) == len(rn) and all(a[c] == b[c] for a, b in zip(ro, rn) for c in cols)
    arrays, diff = [], []
    for fo in sorted((old / "traj").glob("batch_*.npz")):
        a, b = np.load(fo), np.load(new / "traj" / fo.name)
        for key in a.files:
            arrays.append(key)
            if key not in b.files or a[key].dtype != b[key].dtype or a[key].shape != b[key].shape \
                    or a[key].tobytes() != b[key].tobytes():
                diff.append(f"{fo.name}:{key}")
        extra = sorted(set(b.files) - set(a.files))
        tip_ok = "finger_tip" in b.files and b["finger_tip"].shape[2:] == (4, 3) \
            and not np.isnan(b["finger_tip"][b["alive"]]).any()
    n_steps = sum(int(r["steps"]) for r in ro)
    return dict(episodes=len(ro), total_steps=n_steps, success_old=sum(int(r["success"]) for r in ro),
                success_new=sum(int(r["success"]) for r in rn), csv_columns_compared=len(cols),
                csv_identical=csv_same, traj_arrays_compared=sorted(set(arrays)), traj_diff=diff,
                new_extra_arrays=extra, finger_tip_ok=bool(tip_ok),
                passed=bool(csv_same and not diff and tip_ok and extra == ["finger_tip"]))


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if not a or a[0] != "compare" or len(a) < 4:
        raise SystemExit(__doc__)
    old_root, new_root, names = Path(a[1]), Path(a[2]), a[3:]
    res = {n: compare_cell(old_root / n, new_root / n) for n in names}
    cfg = json.loads((new_root / names[0] / "config.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "record_check.json").write_text(json.dumps(dict(gl_renderer=cfg.get("gl_renderer"), cells=res),
                                                      indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# T13 指尖记录前后动作逐位相同检查（CPU 渲染，种子 0–4，B=5）", "",
             f"渲染器：`{cfg.get('gl_renderer')}`。旧代码 = 加指尖记录之前的提交（git archive），新代码 = 当前工作区；两边都开 `--log-traj`。", "",
             "| 条件 | 回合 | 总步数 | 成功 旧/新 | CSV 逐列相同（除墙钟） | 轨迹数组逐字节相同 | 新增数组 | 通过 |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        lines.append(f"| {n} | {r['episodes']} | {r['total_steps']} | {r['success_old']}/{r['success_new']} | "
                     f"{'是' if r['csv_identical'] else '否'}（{r['csv_columns_compared']} 列） | "
                     f"{'是' if not r['traj_diff'] else '否：' + ', '.join(r['traj_diff'][:5])}（{len(r['traj_arrays_compared'])} 个：{', '.join(r['traj_arrays_compared'])}） | "
                     f"{', '.join(r['new_extra_arrays'])} | {'通过' if r['passed'] else '**不通过**'} |")
    ok = all(r["passed"] for r in res.values())
    lines += ["", f"**结论：{'全部通过' if ok else '有不通过'}**。"]
    (OUT / "record_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
