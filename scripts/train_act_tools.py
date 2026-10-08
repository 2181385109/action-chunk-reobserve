"""T10a 自训练 ACT 的纯函数工具（不 import torch / lerobot，单元测试直接测）+ 小命令行。

- train_args()        lerobot 0.3.3 训练脚本的命令行参数（官方超参数；冒烟 / 测速 / 正式训练共用）
- plan_steps()        训练步数规则（统筹定）：官方步数估计 <= 10 小时用官方步数，否则 10 小时能跑完的最大步数，向下取整到 5000 的倍数
- parse_train_log()   从训练日志（可能含多次续训）解析 loss 曲线
- diff_train_config() 与官方 train_config.json 比较

命令行（WSL 内或 Windows 均可，只用标准库 + 可选 matplotlib）：
  python scripts/train_act_tools.py args --steps 100000 --workers 4 --out /root/train_out/act_own
  python scripts/train_act_tools.py plan --sps 5.0 [--startup 30]
  python scripts/train_act_tools.py eta --sps 5.0 --steps 50000 --done 5000
  python scripts/train_act_tools.py parse <train_log.txt> --out-dir results/train_act_own
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_CFG_PATH = ROOT / "docs" / "official_model_cards" / "act_aloha_sim_transfer_cube_human_train_config.json"
OFFICIAL_CFG = json.loads(OFFICIAL_CFG_PATH.read_text(encoding="utf-8"))
REPO_ID = OFFICIAL_CFG["dataset"]["repo_id"]          # lerobot/aloha_sim_transfer_cube_human
OFFICIAL_STEPS = int(OFFICIAL_CFG["steps"])            # 100000
TRAIN_SEED = 1000                                      # 统筹定；与官方 train_config 的 seed 相同
SAVE_FREQ = 5000
LOG_FREQ = 200                                         # 与官方相同
BUDGET_H = 10.0
ROUND_TO = 5000
# 官方用 pyav；本机 lerobot 0.3.3 的 pyav 路径（torchvision 0.22 → av 19）报 TypeError: open() got an unexpected
# keyword argument 'metadata_errors'。torchcodec（lerobot 0.3.3 装了它时的默认后端）与直接用 PyAV 解出的像素逐位相同
# （results/train_act_own/prep/dataset_info.json 的 pyav_vs_torchcodec_pixels），所以用 torchcodec，不降级 av。
VIDEO_BACKEND = "torchcodec"
RESUME_MARK ="### TRAIN_ACT_START step="              # train_act.sh 每次（重新）启动时写进日志


def train_args(steps: int, num_workers: int, output_dir: str, save_freq: int = SAVE_FREQ,
               log_freq: int = LOG_FREQ, seed: int = TRAIN_SEED) -> list[str]:
    """lerobot.scripts.train 的参数。官方超参数大多是 policy.type=act 的默认值，这里把关键的显式写出来；
    其余字段由 diffcfg 与官方 train_config.json 逐项核对。与官方的有意不同：
    eval_freq=0（训练中不做仿真评估，评估用我们自己的 scripts/eval.py）、wandb 关闭、push_to_hub 关闭、
    video_backend（见 VIDEO_BACKEND）、save_freq / steps / num_workers / output_dir / job_name。"""
    oc, op = OFFICIAL_CFG, OFFICIAL_CFG["policy"]
    return [
        "--policy.type=act",
        "--policy.device=cuda",
        f"--policy.use_amp={str(op['use_amp']).lower()}",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={REPO_ID}",
        f"--dataset.video_backend={VIDEO_BACKEND}",
        f"--env.type={oc['env']['type']}",
        f"--env.task={oc['env']['task']}",
        f"--seed={seed}",
        f"--batch_size={oc['batch_size']}",
        f"--steps={steps}",
        f"--num_workers={num_workers}",
        f"--save_freq={save_freq}",
        f"--log_freq={log_freq}",
        "--eval_freq=0",
        "--wandb.enable=false",
        "--job_name=aloha_act_own",
        f"--output_dir={output_dir}",
    ]


def plan_steps(steps_per_s: float, official_steps: int = OFFICIAL_STEPS, budget_h: float = BUDGET_H,
               round_to: int = ROUND_TO, startup_s: float = 0.0) -> dict:
    """统筹定的机械规则。估计耗时 = startup_s + 步数 / 稳态速度。"""
    if steps_per_s <= 0:
        raise ValueError("steps_per_s 必须 > 0")
    est_official_h = (startup_s + official_steps / steps_per_s) / 3600
    if est_official_h <= budget_h:
        steps, rule = official_steps, "官方步数估计 <= 10 小时 → 用官方步数"
    else:
        max_steps = (budget_h * 3600 - startup_s) * steps_per_s
        steps = int(math.floor(max_steps / round_to)) * round_to
        rule = f"官方步数估计 > {budget_h:g} 小时 → 取 {budget_h:g} 小时能跑完的最大步数 {max_steps:.0f}，向下取整到 {round_to} 的倍数"
    return dict(steps_per_s=steps_per_s, startup_s=startup_s, official_steps=official_steps,
                est_official_h=round(est_official_h, 3), budget_h=budget_h, steps=steps,
                est_steps_h=round((startup_s + steps / steps_per_s) / 3600, 3), rule=rule)


def format_big_number(num, precision=0):
    """与 lerobot.utils.utils.format_big_number 相同（日志里的 step 就是这样显示的，例如 1200 -> '1K'）。"""
    for suffix in ["", "K", "M", "B", "T", "Q"]:
        if abs(num) < 1000.0:
            return f"{num:.{precision}f}{suffix}"
        num /= 1000.0
    return num


_KV = re.compile(r"(\w+):(\S+)")


def parse_train_log(text: str, log_freq: int = LOG_FREQ) -> list[dict]:
    """解析 lerobot 训练日志里的 'step:... loss:...' 行。

    lerobot 显示的 step 是取整后的（1200 显示 '1K'），所以真实步数 = 本次启动的起始步数 + 第几行 × log_freq；
    起始步数来自 train_act.sh 写的 '### TRAIN_ACT_START step=N' 标记（没有标记时 = 0）。每行都与显示值核对。
    续训：遇到新标记 step=N 时，丢掉之前记录的 step > N 的行（它们在中断后会从检查点 N 重新训练）。
    返回 [{step, loss, grdn, lr, updt_s, data_s, run}]，按 step 升序。
    """
    recs: list[dict] = []
    cur, run = 0, 0
    for line in text.splitlines():
        if line.startswith(RESUME_MARK):
            cur = int(line[len(RESUME_MARK):].split()[0])
            run += 1
            recs = [r for r in recs if r["step"] <= cur]
            continue
        if " step:" not in line or " loss:" not in line:
            continue
        kv = dict(_KV.findall(line[line.index(" step:"):]))
        cur += log_freq
        if kv["step"] != format_big_number(cur):
            raise ValueError(f"日志步数对不上：推算 {cur}（应显示 {format_big_number(cur)}），日志显示 {kv['step']}：{line!r}")
        recs.append(dict(step=cur, loss=float(kv["loss"]), grdn=float(kv["grdn"]), lr=float(kv["lr"]),
                         updt_s=float(kv["updt_s"]), data_s=float(kv["data_s"]), run=run))
    return recs


def checkpoints_in_log(text: str) -> list[int]:
    return [int(m) for m in re.findall(r"Checkpoint policy after step (\d+)", text)]


def diff_train_config(official: dict, ours: dict, prefix: str = "") -> list[tuple]:
    """递归比较两个 train_config，返回 [(键, 官方值, 我们的值)]（只列不同的）。"""
    rows = []
    for key in sorted(set(official) | set(ours)):
        o, m = official.get(key, "<缺>"), ours.get(key, "<缺>")
        name = f"{prefix}{key}"
        if isinstance(o, dict) and isinstance(m, dict):
            rows += diff_train_config(o, m, name + ".")
        elif o != m:
            rows.append((name, o, m))
    return rows


def eta_hours(steps_per_s: float, total_steps: int, done_steps: int = 0, startup_s: float = 0.0) -> float:
    return (startup_s + max(total_steps - done_steps, 0) / steps_per_s) / 3600


def write_curve(recs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "loss.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "grdn", "lr", "updt_s", "data_s", "run"], lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["step"] for r in recs], [r["loss"] for r in recs], lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"loss (mean over {LOG_FREQ}-step window, log scale)")
    ax.set_title("own ACT on aloha_sim_transfer_cube_human")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("args")
    s.add_argument("--steps", type=int, required=True)
    s.add_argument("--workers", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--save-freq", type=int, default=SAVE_FREQ)
    s.add_argument("--log-freq", type=int, default=LOG_FREQ)
    p = sub.add_parser("plan")
    p.add_argument("--sps", type=float, required=True)
    p.add_argument("--startup", type=float, default=0.0)
    e = sub.add_parser("eta")
    e.add_argument("--sps", type=float, required=True)
    e.add_argument("--steps", type=int, required=True)
    e.add_argument("--done", type=int, default=0)
    e.add_argument("--startup", type=float, default=0.0)
    q = sub.add_parser("parse")
    q.add_argument("log")
    q.add_argument("--out-dir", required=True)
    q.add_argument("--log-freq", type=int, default=LOG_FREQ)
    a = ap.parse_args()
    if a.cmd == "args":
        print("\n".join(train_args(a.steps, a.workers, a.out, a.save_freq, a.log_freq)))
    elif a.cmd == "plan":
        print(json.dumps(plan_steps(a.sps, startup_s=a.startup), indent=2, ensure_ascii=False))
    elif a.cmd == "eta":
        print(f"{eta_hours(a.sps, a.steps, a.done, a.startup):.2f}")
    elif a.cmd == "parse":
        text = Path(a.log).read_text(encoding="utf-8", errors="replace")
        recs = parse_train_log(text, a.log_freq)
        write_curve(recs, Path(a.out_dir))
        last = recs[-1] if recs else None
        print(f"loss 曲线 {len(recs)} 点 -> {a.out_dir}/loss.csv, loss.png；最后：{last}；检查点：{checkpoints_in_log(text)}")


if __name__ == "__main__":
    main()
