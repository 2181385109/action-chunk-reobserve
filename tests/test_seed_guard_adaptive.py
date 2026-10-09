"""T11b：mode=adaptive 的种子保护用 PREREGISTRATION_ADAPTIVE.md。hub ACT / DP / 自训练 ACT 的行为不变。不跑仿真。"""
import functools
import subprocess

import pytest

from chunkre.seed_guard import check_seeds, guard_model, prereg_file, preregistration_committed

CKPT = "/root/train_out/act_own/checkpoints/100000/pretrained_model"


def _git_commit(root, name):
    g = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(g + ["add", name], check=True)
    subprocess.run(g + ["commit", "-q", "-m", name], check=True)


def _repo(tmp_path, names):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for n in names:
        (tmp_path / n).write_text("x\n", encoding="utf-8")
        _git_commit(tmp_path, n)


def test_guard_model_adaptive_keys():
    assert prereg_file("act_adaptive") == "PREREGISTRATION_ADAPTIVE.md"
    assert guard_model("act", None, "adaptive") == "act_adaptive" and guard_model(None, None, "adaptive") == "act_adaptive"
    assert guard_model("act", "", "adaptive") == "act_adaptive"
    assert guard_model("act", CKPT, "adaptive") == "act_adaptive_own"      # 登记没覆盖的组合
    # 其余全部与 T10b 相同（mode 参数可有可无）
    for mode in (None, "fixed", "te"):
        assert guard_model("act", None, mode) == "act" and guard_model(None, None, mode) is None
        assert guard_model("act", CKPT, mode) == "act_own" and guard_model("dp", None, mode) == "dp"
    assert guard_model("act", CKPT) == "act_own" and guard_model("dp", CKPT) == "dp"
    assert guard_model("dp", None, "adaptive") == "dp"                     # adaptive 只对 ACT 有意义（config 另有校验）


def test_adaptive_final_seeds_blocked_without_adaptive_prereg(tmp_path):
    """演示用例：ACT / DP / OWN 的登记都已提交，没有 ADAPTIVE 登记 -> 自适应被拦，其余不受影响。"""
    _repo(tmp_path, ["PREREGISTRATION.md", "PREREGISTRATION_DP.md", "PREREGISTRATION_OWN.md"])
    seeds = range(200000, 200200)
    check_seeds(seeds, root=tmp_path, model=guard_model("act", None, "fixed"))
    check_seeds(seeds, root=tmp_path, model=guard_model("act", None, "te"))
    check_seeds(seeds, root=tmp_path, model="dp")
    check_seeds(seeds, root=tmp_path, model=guard_model("act", CKPT, "fixed"))
    ad = guard_model("act", None, "adaptive")
    with pytest.raises(RuntimeError, match="PREREGISTRATION_ADAPTIVE.md"):
        check_seeds([200000], root=tmp_path, model=ad)
    # 调参种子不拦（试跑用 6000–6049）；保留区间照旧禁止
    check_seeds(range(6000, 6050), root=tmp_path, model=ad)
    with pytest.raises(RuntimeError, match="保留"):
        check_seeds([100000], root=tmp_path, model=ad)
    # 写了未提交：仍拦；提交后放行
    (tmp_path / "PREREGISTRATION_ADAPTIVE.md").write_text("ad\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PREREGISTRATION_ADAPTIVE.md"):
        check_seeds([200000], root=tmp_path, model=ad)
    _git_commit(tmp_path, "PREREGISTRATION_ADAPTIVE.md")
    check_seeds(seeds, root=tmp_path, model=ad)


def test_adaptive_prereg_does_not_unlock_other_models(tmp_path):
    _repo(tmp_path, ["PREREGISTRATION_ADAPTIVE.md"])
    check_seeds([200000], root=tmp_path, model="act_adaptive")
    for m in (None, "act", "dp", "act_own"):
        with pytest.raises(RuntimeError, match="PREREGISTRATION"):
            check_seeds([200000], root=tmp_path, model=m)


def test_adaptive_with_local_checkpoint_never_unlocked(tmp_path):
    """自适应 + 本地检查点没有登记文件：即使所有登记都提交了，最终种子也被拦。"""
    _repo(tmp_path, ["PREREGISTRATION.md", "PREREGISTRATION_DP.md", "PREREGISTRATION_OWN.md", "PREREGISTRATION_ADAPTIVE.md"])
    with pytest.raises(ValueError, match="登记文件"):
        check_seeds([200000], root=tmp_path, model=guard_model("act", CKPT, "adaptive"))
    check_seeds([5], root=tmp_path, model=guard_model("act", CKPT, "adaptive"))   # 调参种子仍可用


def test_run_eval_passes_mode_to_guard(tmp_path, monkeypatch):
    """真正的 run_eval：mode=adaptive -> act_adaptive；fixed -> act；被拦时不建输出目录、不加载模型。"""
    from chunkre import runner
    from chunkre.config import EvalConfig
    calls = []

    def fake_guard(seeds, root=None, model=None):
        calls.append(model)
        raise RuntimeError("blocked-by-test")

    monkeypatch.setattr(runner, "check_seeds", fake_guard)
    out = tmp_path / "o"
    ad = dict(m=10, tau=0.2906, h_cmp=20, min_gap=20)
    for kw in (dict(mode="adaptive", adaptive=ad), dict(mode="fixed", k=100),
               dict(mode="adaptive", adaptive=ad, policy_path="/x/pretrained_model"), dict(policy_path="/x/pretrained_model", k=100)):
        with pytest.raises(RuntimeError, match="blocked-by-test"):
            runner.run_eval(EvalConfig(model="act", seeds=[200000, 200001], batch_size=2, **kw), out)
    assert calls == ["act_adaptive", "act", "act_adaptive_own", "act_own"] and not out.exists()


def test_run_eval_with_real_guard_blocks_adaptive_in_repo_without_prereg(tmp_path, monkeypatch):
    from chunkre import runner
    from chunkre.config import EvalConfig
    subprocess.run(["git", "init", "-q", str(tmp_path / "r")], check=True)
    monkeypatch.setattr(runner, "check_seeds", functools.partial(check_seeds, root=tmp_path / "r"))
    cfg = EvalConfig(model="act", seeds="200000-200009", batch_size=10, mode="adaptive",
                     adaptive=dict(m=10, tau=0.2906, h_cmp=20, min_gap=20))
    with pytest.raises(RuntimeError, match="PREREGISTRATION_ADAPTIVE.md"):
        runner.run_eval(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_repo_adaptive_prereg_is_committed():
    if not preregistration_committed():
        pytest.skip("没有 git 历史")
    assert preregistration_committed(model="act_adaptive")
    check_seeds([200000, 200199], model="act_adaptive")
