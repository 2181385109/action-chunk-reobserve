"""T10b seed_guard 拦截演示：评估自训练 ACT（--policy-path 加载本地检查点）用 >=200000 的种子，
在 PREREGISTRATION_OWN.md 没有提交的仓库里被拦住。在一个临时 git 仓库里做（里面只提交了 ACT 和 DP 的登记，没有 OWN 登记），
不跑任何回合，也不动本仓库。分两层演示：① check_seeds(guard_model(...))；② 真正的 runner.run_eval（把它用的守卫指到临时仓库）。

  wsl.exe -d Ubuntu-24.04 -u root -- bash <repo>/scripts/seed_guard_demo_own.sh
"""
import functools
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chunkre import main_own as mo  # noqa: E402
from chunkre import runner  # noqa: E402
from chunkre.config import EvalConfig, parse_seeds  # noqa: E402
from chunkre.seed_guard import check_seeds, guard_model  # noqa: E402

SEEDS = parse_seeds("200000-200199")


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=demo", "-c", "user.email=demo@demo", *args], check=True,
                   capture_output=True)


def try_guard(label, root, model, policy_path):
    key = guard_model(model, policy_path)
    try:
        check_seeds(SEEDS, root=root, model=key)
        print(f"[{label}] model={model!r} policy_path={'本地检查点' if policy_path else None} (守卫键 {key!r}): 通过（放行）")
        return True
    except RuntimeError as e:
        print(f"[{label}] model={model!r} policy_path={'本地检查点' if policy_path else None} (守卫键 {key!r}): RuntimeError: {e}")
        return False


def try_run_eval(label, root, out):
    """真正的 runner.run_eval：守卫指到临时仓库；被拦住时第一件事就抛错，不建输出目录、不加载模型。"""
    real = runner.check_seeds
    runner.check_seeds = functools.partial(real, root=root)
    try:
        cfg = EvalConfig(model="act", seeds="200000-200009", batch_size=10, k=100, policy_path=mo.CHECKPOINT)
        runner.run_eval(cfg, out)
        print(f"[{label}] run_eval: 通过守卫（不应该在没有登记时发生）")
        return True
    except RuntimeError as e:
        print(f"[{label}] run_eval(--policy-path {Path(mo.CHECKPOINT).name}, seeds 200000-200009): RuntimeError: {e}"
              f"（输出目录已创建：{out.exists()}）")
        return False
    finally:
        runner.check_seeds = real


with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ("PREREGISTRATION.md", "PREREGISTRATION_DP.md"):
        (root / name).write_text(f"{name} (demo)\n", encoding="utf-8")
        git(root, "add", name)
    git(root, "commit", "-q", "-m", "act + dp prereg")
    print("# 临时仓库：已提交 PREREGISTRATION.md（ACT）和 PREREGISTRATION_DP.md，没有 PREREGISTRATION_OWN.md")
    ok_hub = try_guard("hub 预训练 ACT ", root, "act", None)
    ok_dp = try_guard("DP             ", root, "dp", None)
    ok_own = try_guard("自训练 ACT     ", root, "act", mo.CHECKPOINT)
    out = Path(td) / "out_blocked"
    ok_run = try_run_eval("自训练 ACT     ", root, out)
    (root / "PREREGISTRATION_OWN.md").write_text("OWN prereg (demo)\n", encoding="utf-8")
    print("# 写了 PREREGISTRATION_OWN.md 但还没 git commit")
    ok_own2 = try_guard("自训练 ACT     ", root, "act", mo.CHECKPOINT)
    git(root, "add", "PREREGISTRATION_OWN.md")
    git(root, "commit", "-q", "-m", "own prereg")
    print("# 提交 PREREGISTRATION_OWN.md 之后")
    ok_own3 = try_guard("自训练 ACT     ", root, "act", mo.CHECKPOINT)

assert ok_hub and ok_dp and not ok_own and not ok_run and not ok_own2 and ok_own3, "演示结果不符合预期"
print("=== seed_guard 拦截：种子 200000-200199 在 PREREGISTRATION_OWN.md 提交之前不能用于自训练 ACT。没有运行任何回合。")
print("演示通过：hub 预训练 ACT 和 DP 不受影响；自训练 ACT 在 OWN 登记提交前被拦（包括真正的 run_eval），提交后放行。")
