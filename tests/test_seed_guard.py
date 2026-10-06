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


def test_repo_has_no_prereg_yet():
    """仓库当前不应有已提交的 PREREGISTRATION.md，所以 >=200000 在本仓库里必须报错。"""
    with pytest.raises(RuntimeError):
        check_seeds([200000])
