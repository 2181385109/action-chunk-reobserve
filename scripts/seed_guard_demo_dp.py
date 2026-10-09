"""T9 seed_guard 拦截演示：>=200000 的种子 + model="dp"，在 PREREGISTRATION_DP.md 没有提交的仓库里被拦住。
在一个临时 git 仓库里做（里面只提交了 ACT 的 PREREGISTRATION.md，没有 DP 登记），不跑任何回合，也不动本仓库。

  wsl.exe -d Ubuntu-24.04 -u root -- bash <repo>/scripts/seed_guard_demo_dp.sh
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chunkre.config import parse_seeds  # noqa: E402
from chunkre.seed_guard import check_seeds  # noqa: E402

SEEDS = parse_seeds("200000-200199")


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=demo", "-c", "user.email=demo@demo", *args], check=True,
                   capture_output=True)


def try_guard(label, root, model):
    try:
        check_seeds(SEEDS, root=root, model=model)
        print(f"[{label}] model={model!r}: 通过（放行）")
        return True
    except RuntimeError as e:
        print(f"[{label}] model={model!r}: RuntimeError: {e}")
        return False


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "PREREGISTRATION.md").write_text("ACT prereg (demo)\n", encoding="utf-8")
    git(root, "add", "PREREGISTRATION.md")
    git(root, "commit", "-q", "-m", "act prereg")
    print("# 临时仓库：已提交 PREREGISTRATION.md（ACT），没有 PREREGISTRATION_DP.md")
    ok_act = try_guard("ACT", root, "act")
    ok_dp = try_guard("DP ", root, "dp")
    (root / "PREREGISTRATION_DP.md").write_text("DP prereg (demo)\n", encoding="utf-8")
    print("# 写了 PREREGISTRATION_DP.md 但还没 git commit")
    ok_dp2 = try_guard("DP ", root, "dp")
    git(root, "add", "PREREGISTRATION_DP.md")
    git(root, "commit", "-q", "-m", "dp prereg")
    print("# 提交 PREREGISTRATION_DP.md 之后")
    ok_dp3 = try_guard("DP ", root, "dp")

assert ok_act and not ok_dp and not ok_dp2 and ok_dp3, "演示结果不符合预期"
print("=== seed_guard 拦截：种子 200000-200199 在 PREREGISTRATION_DP.md 提交之前不能用于 DP。没有运行任何回合。")
print("演示通过：ACT 不受影响；DP 在 DP 登记提交前被拦，提交后放行。")
