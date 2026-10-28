from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path):
    with (ROOT / path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_phase2_config_is_valid():
    config = load_yaml("configs/phase2.yaml")

    assert config["schema_version"] == 1
    assert config["project"]["phase"] == 2
    assert config["project"]["status"] == "development"
    assert config["phase1"]["preserve_canonical_artifacts"]
    assert not config["phase1"]["allow_rewriting_phase1_results"]


def test_phase2_matches_frozen_risk_contract():
    config = load_yaml("configs/phase2.yaml")
    frozen = load_yaml("configs/model_freeze.yaml")

    assert (
        config["portfolio"]["constituents"]
        == frozen["portfolio"]["constituents"]
    )
    assert (
        config["risk"]["target"]
        == frozen["portfolio"]["forecast_target"]
    )
    assert (
        config["risk"]["quantile_alpha"]
        == frozen["evaluation"]["alpha"]
    )
    assert (
        config["risk"]["horizon_trading_days"]
        == frozen["evaluation"]["forecast_horizon_trading_days"]
    )


def test_phase2_validation_and_registry_isolation():
    config = load_yaml("configs/phase2.yaml")

    validation = config["validation"]
    registry = config["forecast_registry"]

    assert validation["method"] == "expanding_time_series_cv"
    assert validation["n_splits"] == 5
    assert validation["shuffle"] is False
    assert validation["prevent_lookahead"] is True
    assert validation["separate_final_evaluation"] is True

    assert registry["path"].startswith("data/phase2/")
    assert registry["preserve_original_forecasts"] is True
    assert registry["verification_states"] == [
        "PENDING",
        "VERIFIED",
    ]