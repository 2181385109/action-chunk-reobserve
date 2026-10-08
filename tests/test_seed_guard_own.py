"""T10b：自训练 ACT（--policy-path 加载本地检查点）的种子保护用 PREREGISTRATION_OWN.md。不跑仿真。"""
import functools
import subprocess

import pytest

from chunkre.seed_guard import check_seeds, guard_model, prereg_file, preregistration_committed


def _git_commit(root, name):
    g = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(g + ["add", name], check=True)
    subprocess.run(g + ["commit", "-q", "-m", name], check=True)


def test_guard_model_and_prereg_file_own():
    assert prereg_file("act_own") == "PREREGISTRATION_OWN.md"
    ckpt = "/root/train_out/act_own/checkpoints/100000/pretrained_model"
    assert guard_model("act", ckpt) == "act_own" and guard_model(None, ckpt) == "act_own"
    for m, pp in (("act", None), ("act", ""), (None, None), ("dp", None), ("dp", ckpt)):   # 其余照旧
        assert guard_model(m, pp) == m
    # DP 没有自训练登记：本地 DP 检查点仍按 DP 的登记文件
    assert prereg_file(guard_model("dp", ckpt)) == "PREREGISTRATION_DP.md"


def test_own_final_seeds_blocked_without_own_prereg(tmp_path):
    """演示用例：ACT 和 DP 的登记都已提交，没有 OWN 登记 -> 自训练 ACT 被拦，hub ACT / DP 不受影响。"""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in ("PREREGISTRATION.md", "PREREGISTRATION_DP.md"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")
        _git_commit(tmp_path, name)
    own = guard_model("act", "/some/ckpt")
    check_seeds(range(200000, 200200), root=tmp_path, model=guard_model("act", None))
    check_seeds(range(200000, 200200), root=tmp_path, model="dp")
    with pytest.raises(RuntimeError, match="PREREGISTRATION_OWN.md"):
        check_seeds([200000], root=tmp_path, model=own)
    # 调参 / 复现种子（含基础水平测试 1000–1499）这里不拦（run_main_own.sh 另外检查登记）；保留区间照旧禁止
    check_seeds(range(1000, 1500), root=tmp_path, model=own)
    with pytest.raises(RuntimeError, match="保留"):
        check_seeds([100000], root=tmp_path, model=own)
    # 写了未提交：仍拦；提交后放行
    (tmp_path / "PREREGISTRATION_OWN.md").write_text("own\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PREREGISTRATION_OWN.md"):
        check_seeds([200000], root=tmp_path, model=own)
    _git_commit(tmp_path, "PREREGISTRATION_OWN.md")
    check_seeds(range(200000, 200200), root=tmp_path, model=own)


def test_only_own_prereg_does_not_unlock_hub_act(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "PREREGISTRATION_OWN.md").write_text("own\n", encoding="utf-8")
    _git_commit(tmp_path, "PREREGISTRATION_OWN.md")
    check_seeds([200000], root=tmp_path, model="act_own")
    for m in (None, "act", "dp"):
        with pytest.raises(RuntimeError, match="PREREGISTRATION"):
            check_seeds([200000], root=tmp_path, model=m)


def test_run_eval_blocks_local_act_checkpoint_before_anything(tmp_path, monkeypatch):
    """真正的 run_eval：EvalConfig(policy_path=...) 时第一件事用 act_own 守卫；被拦住时不建输出目录、不加载模型。"""
    from chunkre import runner
    from chunkre.config import EvalConfig
    calls = []

    def fake_guard(seeds, root=None, model=None):
        calls.append(model)
        raise RuntimeError("blocked-by-test")

    monkeypatch.setattr(runner, "check_seeds", fake_guard)
    out = tmp_path / "o"
    with pytest.raises(RuntimeError, match="blocked-by-test"):
        runner.run_eval(EvalConfig(model="act", seeds=[200000, 200001], batch_size=2, policy_path="/x/pretrained_model"), out)
    with pytest.raises(RuntimeError, match="blocked-by-test"):
        runner.run_eval(EvalConfig(model="act", seeds=[200000, 200001], batch_size=2), out)
    assert calls == ["act_own", "act"] and not out.exists()


def test_run_eval_with_real_guard_blocks_in_repo_without_own_prereg(tmp_path, monkeypatch):
    """守卫指到没有登记的临时仓库：自训练配置的真实 run_eval 抛出点名 OWN 文件的错，且什么都没创建。"""
    from chunkre import runner
    from chunkre.config import EvalConfig
    subprocess.run(["git", "init", "-q", str(tmp_path / "r")], check=True)
    monkeypatch.setattr(runner, "check_seeds", functools.partial(check_seeds, root=tmp_path / "r"))
    with pytest.raises(RuntimeError, match="PREREGISTRATION_OWN.md"):
        runner.run_eval(EvalConfig(model="act", seeds="200000-200009", batch_size=10, policy_path="/x/pretrained_model"),
                        tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_repo_own_prereg_is_committed():
    if not preregistration_committed():
        pytest.skip("没有 git 历史")
    assert preregistration_committed(model="act_own")
    check_seeds([200000, 200199], model="act_own")
