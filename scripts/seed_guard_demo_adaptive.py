"""T11b seed_guard 拦截演示：mode=adaptive 用 >=200000 的种子，在 PREREGISTRATION_ADAPTIVE.md 没有提交的仓库里被拦住。
在一个临时 git 仓库里做（里面只提交了 ACT / DP / OWN 的登记，没有 ADAPTIVE 登记），不跑任何回合，也不动本仓库。
分两层演示：① check_seeds(guard_model(...))；② 真正的 runner.run_eval（把它用的守卫指到临时仓库，被拦时不建输出目录）。

  wsl.exe -d Ubuntu-24.04 -u root --cd /mnt/d/xiangmu/chunk-reobserve -- bash scripts/seed_guard_demo_adaptive.sh
"""
import functools
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chunkre import runner  # noqa: E402
from chunkre.config import EvalConfig, parse_seeds  # noqa: E402
from chunkre.seed_guard import check_seeds, guard_model  # noqa: E402

SEEDS = parse_seeds("200000-200199")
CKPT = "/root/train_out/act_own/checkpoints/100000/pretrained_model"


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=demo", "-c", "user.email=demo@demo", *args], check=True,
                   capture_output=True)


def try_guard(label, root, model, policy_path=None, mode="fixed"):
    key = guard_model(model, policy_path, mode)
    desc = f"model={model!r} mode={mode!r} policy_path={'本地检查点' if policy_path else None}"
    try:
        check_seeds(SEEDS, root=root, model=key)
        print(f"[{label}] {desc} (守卫键 {key!r}): 通过（放行）")
        return True
    except (RuntimeError, ValueError) as e:
        print(f"[{label}] {desc} (守卫键 {key!r}): {type(e).__name__}: {e}")
        return False


def try_run_eval(label, root, out):
    """真正的 runner.run_eval（mode=adaptive，ACT hub 模型）：守卫指到临时仓库；被拦住时第一件事就抛错，不建输出目录、不加载模型。"""
    real = runner.check_seeds
    runner.check_seeds = functools.partial(real, root=root)
    try:
        cfg = EvalConfig(model="act", seeds="200000-200009", batch_size=10, mode="adaptive",
                         adaptive=dict(m=10, tau=0.2906, h_cmp=20, min_gap=20))
        runner.run_eval(cfg, out)
        print(f"[{label}] run_eval: 通过守卫（不应该在没有登记时发生）")
        return True
    except RuntimeError as e:
        print(f"[{label}] run_eval(mode=adaptive, seeds 200000-200009): RuntimeError: {e}（输出目录已创建：{out.exists()}）")
        return False
    finally:
        runner.check_seeds = real


with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ("PREREGISTRATION.md", "PREREGISTRATION_DP.md", "PREREGISTRATION_OWN.md"):
        (root / name).write_text(f"{name} (demo)\n", encoding="utf-8")
        git(root, "add", name)
    git(root, "commit", "-q", "-m", "act + dp + own prereg")
    print("# 临时仓库：已提交 ACT / DP / OWN 的登记，没有 PREREGISTRATION_ADAPTIVE.md")
    ok_hub = try_guard("hub 预训练 ACT (fixed)   ", root, "act")
    ok_dp = try_guard("DP (fixed)                ", root, "dp")
    ok_own = try_guard("自训练 ACT (fixed)       ", root, "act", CKPT)
    ok_ad = try_guard("自适应 (adaptive)         ", root, "act", None, "adaptive")
    ok_ad_own = try_guard("自适应 + 自训练检查点    ", root, "act", CKPT, "adaptive")
    out = Path(td) / "out_blocked"
    ok_run = try_run_eval("自适应 (adaptive)         ", root, out)
    (root / "PREREGISTRATION_ADAPTIVE.md").write_text("ADAPTIVE prereg (demo)\n", encoding="utf-8")
    print("# 写了 PREREGISTRATION_ADAPTIVE.md 但还没 git commit")
    ok_ad2 = try_guard("自适应 (adaptive)         ", root, "act", None, "adaptive")
    git(root, "add", "PREREGISTRATION_ADAPTIVE.md")
    git(root, "commit", "-q", "-m", "adaptive prereg")
    print("# 提交 PREREGISTRATION_ADAPTIVE.md 之后")
    ok_ad3 = try_guard("自适应 (adaptive)         ", root, "act", None, "adaptive")
    ok_ad_own2 = try_guard("自适应 + 自训练检查点    ", root, "act", CKPT, "adaptive")

assert ok_hub and ok_dp and ok_own, "hub ACT / DP / 自训练 ACT 不应受影响"
assert not ok_ad and not ok_run and not ok_ad2 and ok_ad3, "演示结果不符合预期"
assert not ok_ad_own and not ok_ad_own2, "自适应 + 本地检查点没有登记，始终应被拦"
print("=== seed_guard 拦截：种子 200000-200199 在 PREREGISTRATION_ADAPTIVE.md 提交之前不能用于自适应规则。没有运行任何回合。")
print("演示通过：hub 预训练 ACT / DP / 自训练 ACT 不受影响；自适应在 ADAPTIVE 登记提交前被拦（包括真正的 run_eval），提交后放行；"
      "自适应 + 本地检查点（登记没覆盖）始终被拦。")
