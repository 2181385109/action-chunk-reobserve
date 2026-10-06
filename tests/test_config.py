import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkre.config import EvalConfig, k_upper_bound, load_config, parse_seeds, validate_k

CARDS = Path(__file__).resolve().parent.parent / "docs" / "official_model_cards"


def hub_cfg(name):
    d = json.loads((CARDS / name).read_text(encoding="utf-8"))
    return SimpleNamespace(**d)


ACT = hub_cfg("act_aloha_sim_transfer_cube_human_config.json")
DP = hub_cfg("diffusion_pusht_config.json")


def test_upper_bounds_from_hub_configs():
    assert ACT.n_action_steps == 100 and ACT.chunk_size == 100
    assert DP.n_action_steps == 8 and DP.horizon == 16 and DP.n_obs_steps == 2
    assert k_upper_bound("act", ACT) == 100
    assert k_upper_bound("dp", DP) == 15  # horizon - n_obs_steps + 1


@pytest.mark.parametrize("k", [1, 2, 50, 100])
def test_act_k_ok(k):
    validate_k("act", "fixed", k, ACT)


@pytest.mark.parametrize("k", [0, 101, 200, -1])
def test_act_k_out_of_range(k):
    with pytest.raises(ValueError, match="1..100"):
        validate_k("act", "fixed", k, ACT)


@pytest.mark.parametrize("k", [1, 8, 15])
def test_dp_k_ok(k):
    validate_k("dp", "fixed", k, DP)


@pytest.mark.parametrize("k", [0, 16, 100])
def test_dp_k_out_of_range(k):
    with pytest.raises(ValueError, match="1..15"):
        validate_k("dp", "fixed", k, DP)


def test_te_rules():
    validate_k("act", "te", 1, ACT)
    with pytest.raises(ValueError):
        validate_k("act", "te", 5, ACT)
    with pytest.raises(ValueError):
        validate_k("dp", "te", 1, DP)
    c = EvalConfig(model="act", seeds="0-9", mode="te", k=None)
    assert c.k_effective == 1


def test_parse_seeds():
    assert parse_seeds("0-9") == list(range(10))
    assert parse_seeds("1000-1499") == list(range(1000, 1500))
    assert parse_seeds("0,3,5") == [0, 3, 5]
    assert parse_seeds([1, 2]) == [1, 2]
    assert parse_seeds({"start": 5, "n": 3}) == [5, 6, 7]


def test_eval_config_defaults_and_batches(tmp_path):
    c = EvalConfig(model="dp", seeds="0-9", batch_size=5)
    assert c.k_effective == 8 and c.batches() == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert EvalConfig(model="act", seeds="0-9").k_effective == 100
    with pytest.raises(ValueError, match="整数倍"):
        EvalConfig(model="act", seeds="0-9", batch_size=3)
    y = tmp_path / "c.yaml"
    y.write_text("model: act\nseeds: 0-19\nbatch_size: 10\n", encoding="utf-8")
    c = load_config(y, dict(k=10, batch_size=None))
    assert c.k == 10 and c.batch_size == 10 and len(c.seeds) == 20
    y.write_text("model: act\nseeds: 0-9\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知字段"):
        load_config(y)
