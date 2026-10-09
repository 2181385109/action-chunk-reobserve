"""把复现比较结果 results/repro/compare_official.json 写进 docs/ENVIRONMENT.md 官方基线小节（T5）。

数字全部从 JSON 读（不手抄）；只替换 ENVIRONMENT.md 里 <!-- repro:begin --> 与 <!-- repro:end --> 之间的内容
（没有标记时插在第 5 节“要点”列表之后、第 6 节之前）。可重复运行。

用法：python scripts/repro_table.py [--check]   （--check：只检查文件里的块是否与 JSON 一致，不写）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "repro" / "compare_official.json"
DOC = ROOT / "docs" / "ENVIRONMENT.md"
BEGIN, END = "<!-- repro:begin -->", "<!-- repro:end -->"
NAMES = {"act": "ACT `act_aloha_sim_transfer_cube_human`", "dp": "DP `diffusion_pusht`"}


def block(res: dict) -> str:
    L = [BEGIN, "",
         "### 复现结果（T2 脚本 `scripts/run_repro.sh`，2026-10-07 手动运行完成；由 `scripts/repro_table.py` 从 "
         "`results/repro/compare_official.json` 生成，勿手改）", "",
         "设置：官方种子 1000–1499 各 500 回合，官方默认推理设置（ACT k=100 无时间集成、DP k=8），GPU 渲染（D3D12）；"
         "逐回合结果 `results/repro/{act,dp}/episodes.csv`。", "",
         "| 模型 | 我们的成功率 | 95% 置信区间（Wilson） | 官方 | 差（百分点） | 判定 | 逐种子一致率（参考） |",
         "|---|---|---|---|---|---|---|"]
    for m in ("act", "dp"):
        r = res[m]
        lo, hi = r["ours_ci95_pc"]
        ps = r.get("per_seed") or {}
        agree = (f"{ps['agreement_rate'] * 100:.1f}%（若两边独立约 {ps['chance_agreement_if_independent'] * 100:.1f}%）"
                 if ps else "—")
        L.append(f"| {NAMES[m]} | {r['ours_pc']:.1f}%（{r['n_episodes']} 回合） | {lo:.1f}–{hi:.1f} | "
                 f"{r['official_pc']:.1f}% | {r['diff_pp']:+.1f} | **{r['verdict']}** | {agree} |")
    L += ["", f"判定标准（看结果前预先定好）：{res['criteria']}。两项都**合格**。",
          "逐种子一致率只作参考：ACT 远高于“独立”时的期望，说明同一种子的成败与官方高度一致（渲染器差异影响很小）；"
          "DP 与独立期望相当，即单个种子的成败与官方基本不配对（可能因为两边扩散去噪抽到的随机噪声不同；未深究）。", "", END]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = json.loads(SRC.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    new_block = block(res)
    if BEGIN in doc:
        pre, rest = doc.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        new = pre + new_block + post
    else:
        anchor = "\n## 6. "
        i = doc.index(anchor)
        new = doc[:i].rstrip("\n") + "\n\n" + new_block + "\n" + doc[i:]
    if a.check:
        ok = new == doc
        print("ENVIRONMENT.md 复现块与 JSON 一致" if ok else "ENVIRONMENT.md 复现块与 JSON 不一致")
        sys.exit(0 if ok else 1)
    DOC.write_text(new, encoding="utf-8", newline="\n")
    print(new_block)


if __name__ == "__main__":
    main()
