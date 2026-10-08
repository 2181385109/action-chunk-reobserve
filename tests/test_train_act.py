"""T10a：自训练 ACT 工具（scripts/train_act_tools.py）+ eval.py --policy-path。不跑训练、不跑仿真。"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_act_tools as T  # noqa: E402
from chunkre.config import EvalConfig, check_local_policy_dir, load_config  # noqa: E402


# ---------------------------------------------------------------- 训练参数
def _argdict(args):
    return dict(a[2:].split("=", 1) for a in args)


def test_train_args_official_hyperparams_and_seed():
    d = _argdict(T.train_args(steps=100000, num_workers=4, output_dir="/root/train_out/act_own"))
    assert d["seed"] == "1000" == str(T.TRAIN_SEED)
    assert d["batch_size"] == "8" == str(T.OFFICIAL_CFG["batch_size"])
    assert d["policy.type"] == "act"
    assert d["policy.use_amp"] == "false"
    assert d["dataset.repo_id"] == "lerobot/aloha_sim_transfer_cube_human"
    assert d["env.task"] == "AlohaTransferCube-v0"
    assert d["save_freq"] == "5000" and d["log_freq"] == "200"
    assert d["eval_freq"] == "0" and d["wandb.enable"] == "false" and d["policy.push_to_hub"] == "false"
    assert d["steps"] == "100000" and d["num_workers"] == "4"
    assert d["output_dir"] == "/root/train_out/act_own"
    assert d["dataset.video_backend"] == "torchcodec"


def test_official_constants():
    assert T.OFFICIAL_STEPS == 100000
    assert T.OFFICIAL_CFG["seed"] == 1000
    assert T.OFFICIAL_CFG["policy"]["optimizer_lr"] == 1e-5


# ---------------------------------------------------------------- 步数规则
def test_plan_official_when_fits():
    p = T.plan_steps(5.0)                     # 100000 / 5 = 20000 s = 5.56 h
    assert p["steps"] == 100000 and p["est_official_h"] == pytest.approx(5.556, abs=1e-3)


def test_plan_capped_rounds_down_to_5000():
    p = T.plan_steps(2.0)                     # 10 h = 72000 步 -> 70000
    assert p["steps"] == 70000
    p = T.plan_steps(2.2)                     # 79200 -> 75000
    assert p["steps"] == 75000
    assert p["est_steps_h"] <= 10


def test_plan_boundary_exactly_10h_uses_official():
    p = T.plan_steps(100000 / 36000)          # 恰好 10 小时
    assert p["steps"] == 100000


def test_plan_startup_counts():
    sps = 100000 / 36000
    assert T.plan_steps(sps, startup_s=60)["steps"] < 100000


def test_plan_rejects_bad_speed():
    with pytest.raises(ValueError):
        T.plan_steps(0)


def test_eta():
    assert T.eta_hours(5.0, 100000, 0) == pytest.approx(20000 / 3600)
    assert T.eta_hours(5.0, 100000, 40000, startup_s=36) == pytest.approx((36 + 12000) / 3600)
    assert T.eta_hours(5.0, 100, 200) == 0


# ---------------------------------------------------------------- 日志解析
def _line(step, loss):
    return (f"INFO 2026-10-08 17:40:09 ts/train.py:232 step:{T.format_big_number(step)} smpl:1K ep:0 epch:0.01 "
            f"loss:{loss} grdn:450.767 lr:1.0e-05 updt_s:0.292 data_s:0.023")


def test_format_big_number_matches_lerobot_display():
    assert T.format_big_number(200) == "200"
    assert T.format_big_number(1200) == "1K"
    assert T.format_big_number(1600) == "2K"
    assert T.format_big_number(100000) == "100K"


def test_parse_simple():
    text = "\n".join(["noise", _line(200, 5.0), _line(400, 4.0), "INFO Checkpoint policy after step 400"])
    recs = T.parse_train_log(text, log_freq=200)
    assert [r["step"] for r in recs] == [200, 400]
    assert recs[1]["loss"] == 4.0 and recs[0]["data_s"] == 0.023
    assert T.checkpoints_in_log(text) == [400]


def test_parse_resume_drops_steps_after_checkpoint():
    lines = [T.RESUME_MARK + "0"] + [_line(s, 10 - s / 200) for s in (200, 400, 600, 800)]
    # 检查点在 400 步，第一次运行跑到 800 后中断；第二次从 400 续训，600、800 重新记录
    lines += [T.RESUME_MARK + "400 resume"] + [_line(s, 1.0) for s in (600, 800, 1000)]
    recs = T.parse_train_log("\n".join(lines), log_freq=200)
    assert [r["step"] for r in recs] == [200, 400, 600, 800, 1000]
    assert [r["run"] for r in recs] == [1, 1, 2, 2, 2]
    assert recs[2]["loss"] == 1.0                       # 续训后的值替换了中断前的


def test_parse_detects_step_mismatch():
    text = "\n".join([_line(200, 1.0), _line(600, 1.0)])   # 少了 400 那行
    with pytest.raises(ValueError, match="对不上"):
        T.parse_train_log(text, log_freq=200)


def test_write_curve(tmp_path):
    recs = T.parse_train_log("\n".join([_line(200, 5.0), _line(400, 4.0)]), log_freq=200)
    T.write_curve(recs, tmp_path)
    rows = (tmp_path / "loss.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "step,loss,grdn,lr,updt_s,data_s,run" and rows[2].startswith("400,4.0,")


def test_diff_train_config():
    a = {"x": 1, "p": {"lr": 1e-5, "n": 2}, "only_a": 0}
    b = {"x": 1, "p": {"lr": 1e-4, "n": 2}, "only_b": 3}
    rows = T.diff_train_config(a, b)
    assert ("p.lr", 1e-5, 1e-4) in rows and ("only_a", 0, "<缺>") in rows and ("only_b", "<缺>", 3) in rows
    assert len(rows) == 3


# 冒烟训练保存的 train_config 与官方的差别（scripts/train_act_prep.sh diffcfg 写出）只允许是有意改的字段
ALLOWED_DIFF = {"steps", "save_freq", "eval_freq", "log_freq", "output_dir", "job_name", "num_workers",
                "wandb.enable", "dataset.video_backend", "policy.push_to_hub"}


def test_smoke_train_config_diff_only_intended():
    f = ROOT / "results" / "train_act_own" / "prep" / "train_config_diff.json"
    if not f.exists():
        pytest.skip("还没有 diffcfg 结果")
    rows = json.loads(f.read_text(encoding="utf-8"))
    # 版本间的字段增减（一边没有这个字段、另一边是空 / 默认值），不是超参数
    schema_only = {r["key"] for r in rows if "<缺>" in (r["official"], r["ours"])
                   and {r["official"], r["ours"]} <= {"<缺>", None, False}}
    unexpected = {r["key"] for r in rows} - ALLOWED_DIFF - schema_only
    assert not unexpected, unexpected
    assert {"seed", "batch_size", "num_workers", "optimizer.lr"}.isdisjoint({r["key"] for r in rows})


# ---------------------------------------------------------------- eval.py --policy-path
def _fake_ckpt(d: Path, typ="act"):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"type": typ, "n_action_steps": 100}), encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"x")
    return d


def test_policy_path_default_none_and_not_in_fingerprint():
    from chunkre.runner import _fingerprint

    a = EvalConfig(model="act", seeds="0-9")
    assert a.policy_path is None
    assert "policy_path" not in _fingerprint(a)
    b = EvalConfig(model="act", seeds="0-9", policy_path="/root/train_out/x/pretrained_model")
    assert _fingerprint(b)["policy_path"] == "/root/train_out/x/pretrained_model"
    assert _fingerprint(a) != _fingerprint(b)


def test_old_config_fingerprint_still_matches():
    """T10a 之前写的 config.json（hub 模型）续跑指纹不变。"""
    from chunkre.runner import _fingerprint

    old = json.loads((ROOT / "results" / "main" / "k100__none" / "config.json").read_text(encoding="utf-8"))
    fp = old["eval_config_fingerprint"]
    cfg = EvalConfig(**{k: v for k, v in fp.items() if k in EvalConfig.__dataclass_fields__})
    assert _fingerprint(cfg) == fp


def test_check_local_policy_dir(tmp_path):
    ok = _fake_ckpt(tmp_path / "ok")
    assert check_local_policy_dir(ok, "act")["type"] == "act"
    with pytest.raises(ValueError, match="不符"):
        check_local_policy_dir(ok, "dp")
    with pytest.raises(FileNotFoundError, match="不是目录"):
        check_local_policy_dir(tmp_path / "nope", "act")
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        check_local_policy_dir(tmp_path / "half", "act")
    assert check_local_policy_dir(_fake_ckpt(tmp_path / "dp", "diffusion"), "dp")["type"] == "diffusion"


def test_eval_cli_passes_policy_path(monkeypatch, tmp_path):
    import chunkre.runner as R

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval as E

    seen = {}
    monkeypatch.setattr(R, "run_eval", lambda cfg, out: seen.update(cfg=cfg, out=out) or {})
    E.main(["--config", str(ROOT / "configs" / "act_default.yaml"), "--out", str(tmp_path / "o"),
            "--seeds", "0-1", "--batch-size", "2", "--k", "100", "--policy-path", "/root/train_out/x/pretrained_model"])
    assert seen["cfg"].policy_path == "/root/train_out/x/pretrained_model"
    E.main(["--config", str(ROOT / "configs" / "act_default.yaml"), "--out", str(tmp_path / "o"),
            "--seeds", "0-1", "--batch-size", "2"])
    assert seen["cfg"].policy_path is None


def test_load_config_policy_path_override():
    c = load_config(None, {"model": "act", "seeds": "0-1", "batch_size": 2, "policy_path": "p"})
    assert c.policy_path == "p"
    assert load_config(None, {"model": "act", "seeds": "0-1", "batch_size": 2, "policy_path": ""}).policy_path is None
