import subprocess

import pytest

from chunkre.seed_guard import DEBUG_MAX, FINAL_MIN, check_seeds, seed_zone


def test_zone_boundaries():
    assert (DEBUG_MAX, FINAL_MIN) == (9999, 200000)
    assert seed_zone(0) == "debug"
    assert seed_zone(9999) == "debug"
    assert seed_zone(10000) == "reserved"
    assert seed_zone(100000) == "reserved"  # 官方 DP 训练种子，保留
    assert seed_zone(199999) == "reserved"
    assert seed_zone(200000) == "final"
    assert seed_zone(-1) == "invalid"


def test_debug_seeds_ok(tmp_path):
    check_seeds([0, 1, 1000, 1499, 9999], root=tmp_path)


@pytest.mark.parametrize("s", [10000, 100000, 100001, 199999])
def test_reserved_seeds_raise(tmp_path, s):
    with pytest.raises(RuntimeError, match="保留"):
        check_seeds([0, s], root=tmp_path)


def test_negative_seed_raises(tmp_path):
    with pytest.raises(RuntimeError):
        check_seeds([-3], root=tmp_path)


def test_final_seeds_need_committed_prereg(tmp_path):
    # 没有 PREREGISTRATION.md
    with pytest.raises(RuntimeError, match="PREREGISTRATION"):
        check_seeds([200000], root=tmp_path)
    # 有文件但未提交
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "PREREGISTRATION.md").write_text("plan\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PREREGISTRATION"):
        check_seeds([250000], root=tmp_path)
    # 提交后允许
    g = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(g + ["add", "PREREGISTRATION.md"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "prereg"], check=True)
    check_seeds([200000, 250000, 5], root=tmp_path)
    # 保留区间仍然禁止
    with pytest.raises(RuntimeError, match="保留"):
        check_seeds([150000], root=tmp_path)


def test_repo_prereg_is_committed_so_final_seeds_pass():
    """T6b 起：PREREGISTRATION.md 已提交（commit 6c5302c），本仓库里 >=200000 的种子通过检查（只是检查，不跑回合）。
    没有已提交登记的目录里仍然报错：见上面的 tmp_path 测试。"""
    from chunkre.seed_guard import preregistration_committed
    if not preregistration_committed():
        pytest.skip("没有 git 历史 / 登记文件未提交")
    check_seeds([200000, 200199])


# ---------- T9：DP 用自己的登记文件 ----------
def _git_commit(root, name):
    g = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(g + ["add", name], check=True)
    subprocess.run(g + ["commit", "-q", "-m", name], check=True)


def test_prereg_file_per_model():
    from chunkre.seed_guard import prereg_file
    assert prereg_file(None) == prereg_file("act") == "PREREGISTRATION.md"
    assert prereg_file("dp") == "PREREGISTRATION_DP.md"
    with pytest.raises(ValueError):
        prereg_file("pi0")


def test_dp_final_seeds_need_dp_prereg(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    # 没有任何登记文件：ACT、DP 都拦
    for m in (None, "act", "dp"):
        with pytest.raises(RuntimeError, match="PREREGISTRATION"):
            check_seeds([200000], root=tmp_path, model=m)
    # 只提交了 ACT 的登记：ACT 放行，DP 仍被拦，报错点名 DP 文件
    (tmp_path / "PREREGISTRATION.md").write_text("act\n", encoding="utf-8")
    _git_commit(tmp_path, "PREREGISTRATION.md")
    check_seeds([200000], root=tmp_path, model="act")
    check_seeds([200000], root=tmp_path)
    with pytest.raises(RuntimeError, match="PREREGISTRATION_DP.md"):
        check_seeds([200000, 200199], root=tmp_path, model="dp")
    # DP 登记文件存在但未提交：仍拦
    (tmp_path / "PREREGISTRATION_DP.md").write_text("dp\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PREREGISTRATION_DP.md"):
        check_seeds([200000], root=tmp_path, model="dp")
    # 提交后放行
    _git_commit(tmp_path, "PREREGISTRATION_DP.md")
    check_seeds([200000, 200199], root=tmp_path, model="dp")
    # 调参区间不需要登记，保留区间照旧禁止
    check_seeds([5000, 5049], root=tmp_path, model="dp")
    with pytest.raises(RuntimeError, match="保留"):
        check_seeds([100000], root=tmp_path, model="dp")


def test_dp_only_dp_prereg_does_not_unlock_act(tmp_path):
    """反过来：只有 DP 登记时，ACT（含 model=None）仍被拦。"""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "PREREGISTRATION_DP.md").write_text("dp\n", encoding="utf-8")
    _git_commit(tmp_path, "PREREGISTRATION_DP.md")
    check_seeds([200000], root=tmp_path, model="dp")
    for m in (None, "act"):
        with pytest.raises(RuntimeError, match="PREREGISTRATION.md"):
            check_seeds([200000], root=tmp_path, model=m)


def test_run_eval_passes_model_to_seed_guard():
    """runner.run_eval 必须把 cfg.model 传给 check_seeds（否则 DP 会按 ACT 的文件放行）。"""
    import inspect

    from chunkre import runner
    src = inspect.getsource(runner.run_eval)
    assert "check_seeds(cfg.seeds, model=cfg.model)" in src


def test_repo_dp_prereg_is_committed():
    from chunkre.seed_guard import preregistration_committed
    if not preregistration_committed():
        pytest.skip("没有 git 历史")
    assert preregistration_committed(model="dp")
    check_seeds([200000, 200199], model="dp")


def test_run_eval_calls_guard_with_model_before_anything_else(tmp_path, monkeypatch):
    """行为检查（不跑回合）：run_eval 的第一件事是 check_seeds(seeds, model=cfg.model)；守卫抛错时什么都没创建。"""
    from chunkre import runner
    from chunkre.config import EvalConfig

    calls = []

    def fake_guard(seeds, root=None, model=None):
        calls.append((list(seeds)[:2], model))
        raise RuntimeError("blocked-by-test")

    monkeypatch.setattr(runner, "check_seeds", fake_guard)
    for model in ("dp", "act"):
        out = tmp_path / model
        with pytest.raises(RuntimeError, match="blocked-by-test"):
            runner.run_eval(EvalConfig(model=model, seeds=[200000, 200001], batch_size=2), out)
        assert not out.exists()
    assert calls == [([200000, 200001], "dp"), ([200000, 200001], "act")]
