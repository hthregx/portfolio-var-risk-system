from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import yaml

EXPECTED_FEATURES = [
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "rolling_vol_5",
    "rolling_vol_20",
    "rolling_vol_60",
    "drawdown",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def assert_close(actual, expected: float, label: str, tol: float = 1e-12) -> None:
    try:
        actual_float = float(actual)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label}: expected numeric {expected!r}, got {actual!r}") from exc
    if abs(actual_float - expected) > tol:
        fail(f"{label}: expected {expected!r}, got {actual_float!r}")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        fail(f"Expected mapping in {path}")
    return data


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        fail(f"Expected object in {path}")
    return data


def source_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def parse_ast(path: Path) -> ast.Module:
    return ast.parse(source_text(path), filename=str(path))


def literal_assignments(module: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return values


def assert_source_contains(path: Path, snippets: list[str]) -> None:
    text = source_text(path)
    for snippet in snippets:
        if snippet not in text:
            fail(f"{path.name}: required implementation evidence not found: {snippet!r}")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "configs" / "model_freeze.yaml").exists():
        # Allow running the downloaded script before moving it into repo/scripts.
        candidate = Path.cwd()
        if (candidate / "configs" / "model_freeze.yaml").exists():
            repo = candidate
        else:
            fail("Run from repository root or place this script under repository/scripts/.")

    freeze = load_yaml(repo / "configs" / "model_freeze.yaml")
    final_cfg = load_yaml(repo / "configs" / "final_evaluation.yaml")
    metadata = load_json(repo / "results" / "final_run_metadata.json")

    f_models = freeze["models"]
    c_hist = final_cfg["historical_simulation"]
    c_ewma = final_cfg["ewma"]
    c_gb = final_cfg["gradient_boosting"]
    m_methods = metadata["methods"]

    # Historical contract.
    for source_name, cfg in [
        ("model_freeze", f_models["historical_simulation"]),
        ("final_evaluation", c_hist),
        ("final_run_metadata", m_methods["historical_simulation"]),
    ]:
        assert_close(cfg["alpha"], 0.05, f"Historical alpha [{source_name}]")
        assert_equal(int(cfg["window"]), 250, f"Historical window [{source_name}]")
        assert_equal(str(cfg["mode"]), "rolling", f"Historical mode [{source_name}]")

    # EWMA contract.
    for source_name, cfg in [
        ("model_freeze", f_models["ewma"]),
        ("final_evaluation", c_ewma),
        ("final_run_metadata", m_methods["ewma"]),
    ]:
        assert_close(cfg["alpha"], 0.05, f"EWMA alpha [{source_name}]")
        assert_close(cfg["decay"], 0.94, f"EWMA decay [{source_name}]")
        assert_equal(str(cfg["mode"]), "expanding", f"EWMA mode [{source_name}]")
        assert_equal(str(cfg["distribution"]).lower(), "normal", f"EWMA distribution [{source_name}]")
        assert_equal(str(cfg["mean_assumption"]).lower(), "zero", f"EWMA mean [{source_name}]")
        assert_equal(str(cfg["initialization"]), "first_squared_return", f"EWMA initialization [{source_name}]")

    # Important narrative contract: 0.90 had stronger validation evidence but was not adopted.
    assert_close(
        f_models["ewma"]["validation_selected_alternative"]["decay"],
        0.90,
        "EWMA validation alternative",
    )
    assert_equal(
        f_models["ewma"]["validation_selected_alternative"]["adopted_for_canonical_evaluation"],
        False,
        "EWMA 0.90 canonical adoption",
    )

    # Gradient Boosting canonical G04 contract.
    expected_gb = {
        "experiment_id": "G04",
        "loss": "quantile",
        "alpha": 0.05,
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 5,
        "subsample": 1.0,
        "random_state": 42,
        "training_protocol": "expanding_refit_each_target",
    }

    for source_name, cfg in [
        ("model_freeze", f_models["gradient_boosting"]),
        ("final_evaluation", c_gb),
        ("final_run_metadata", m_methods["gradient_boosting"]),
    ]:
        for key, expected in expected_gb.items():
            actual = cfg[key]
            if isinstance(expected, float):
                assert_close(actual, expected, f"GB {key} [{source_name}]")
            else:
                assert_equal(actual, expected, f"GB {key} [{source_name}]")
        assert_equal(list(cfg["features"]), EXPECTED_FEATURES, f"GB exact features/order [{source_name}]")

    assert_equal(int(f_models["gradient_boosting"]["feature_count"]), 7, "GB frozen feature_count")
    assert_equal(
        bool(f_models["gradient_boosting"]["market_features_used_by_canonical_g04"]),
        False,
        "GB canonical market_features",
    )
    assert_equal(
        bool(freeze["reproducibility"]["canonical_runner_uses_market_feature_builder"]),
        False,
        "Canonical runner market feature builder",
    )

    # Verify the canonical final config explicitly excludes market/liquidity families.
    excluded = set(c_gb.get("excluded_unimplemented_feature_families", []))
    for family in ["price_range", "trading_volume", "market_index"]:
        if family not in excluded:
            fail(f"GB final_evaluation does not explicitly exclude {family!r}")

    # Source-code implementation evidence.
    historical_path = repo / "src" / "models" / "historical_var.py"
    ewma_path = repo / "src" / "models" / "ewma_var.py"
    gb_path = repo / "src" / "models" / "gradient_boosting_var.py"
    features_path = repo / "src" / "gb_return_features.py"
    runner_path = repo / "scripts" / "run_final_walk_forward.py"

    assert_source_contains(
        historical_path,
        [
            'np.quantile(values, alpha_value, method="linear")',
            'var_value = max(0.0, -quantile_return)',
        ],
    )

    assert_source_contains(
        ewma_path,
        [
            'variance = float(values[0] ** 2)',
            'variance = decay_value * variance + (1.0 - decay_value) * float(value ** 2)',
            'NormalDist().inv_cdf(alpha_value) * volatility',
            'var_value = max(0.0, -quantile_return)',
        ],
    )

    gb_assignments = literal_assignments(parse_ast(gb_path))
    # The class default is intentionally the old prototype 0.05; canonical G04 is injected by config.
    assert_close(gb_assignments["DEFAULT_ALPHA"], 0.05, "GB class default alpha")
    assert_equal(gb_assignments["DEFAULT_N_ESTIMATORS"], 100, "GB class default n_estimators")
    assert_close(gb_assignments["DEFAULT_LEARNING_RATE"], 0.05, "GB class prototype default learning_rate")
    assert_equal(gb_assignments["DEFAULT_MAX_DEPTH"], 2, "GB class default max_depth")
    assert_equal(gb_assignments["DEFAULT_MIN_SAMPLES_LEAF"], 5, "GB class default min_samples_leaf")
    assert_close(gb_assignments["DEFAULT_SUBSAMPLE"], 1.0, "GB class default subsample")
    assert_equal(gb_assignments["DEFAULT_RANDOM_STATE"], 42, "GB class default random_state")

    assert_source_contains(
        gb_path,
        [
            'loss="quantile"',
            'alpha=self.alpha',
            'learning_rate=self.learning_rate',
            'random_state=self.random_state',
        ],
    )

    feature_assignments = literal_assignments(parse_ast(features_path))
    assert_equal(tuple(feature_assignments["RETURN_LAGS"]), (1, 2, 5), "GB RETURN_LAGS implementation")
    assert_equal(tuple(feature_assignments["VOLATILITY_WINDOWS"]), (5, 20, 60), "GB VOLATILITY_WINDOWS implementation")
    assert_source_contains(
        features_path,
        [
            'features[f"return_lag_{lag}"]',
            'available_history = validated_returns.shift(1)',
            'features[f"rolling_vol_{window}"]',
            'features["drawdown"]',
            'contemporaneous_drawdown.shift(1)',
        ],
    )

    # Verify runner injects final config instead of relying on the GB class prototype defaults.
    assert_source_contains(
        runner_path,
        [
            'window = int(model_config["window"])',
            'mode = str(model_config["mode"])',
            'decay = float(model_config["decay"])',
            'feature_columns = list(model_config["features"])',
            'learning_rate=float(model_config["learning_rate"])',
            'min_samples_leaf=int(model_config["min_samples_leaf"])',
            'random_state=int(model_config["random_state"])',
            'gb_frame.index < target_date',
        ],
    )

    # gradient_boosting.yaml is prototype G01 and must NOT be treated as canonical G04.
    prototype = load_yaml(repo / "configs" / "gradient_boosting.yaml")
    assert_equal(prototype["experiment_id"], "G01", "Prototype GB experiment id")
    assert_close(prototype["learning_rate"], 0.05, "Prototype GB learning rate")
    assert_equal(prototype["selection_status"], "prototype_pre_validation", "Prototype GB selection status")

    print("PASS - B28.2 FROZEN SPECIFICATION AUDIT")
    print()
    print("Historical Simulation")
    print("  [PASS] alpha = 0.05")
    print("  [PASS] window = 250")
    print("  [PASS] mode = rolling")
    print()
    print("EWMA")
    print("  [PASS] alpha = 0.05")
    print("  [PASS] decay = 0.94")
    print("  [PASS] mode = expanding")
    print("  [PASS] zero-mean Normal")
    print("  [PASS] initialization = first_squared_return")
    print("  [PASS] decay 0.90 is validation evidence only; not canonical")
    print()
    print("Gradient Boosting")
    print("  [PASS] experiment_id = G04")
    print("  [PASS] alpha = 0.05")
    print("  [PASS] loss = quantile")
    print("  [PASS] n_estimators = 100")
    print("  [PASS] learning_rate = 0.03")
    print("  [PASS] max_depth = 2")
    print("  [PASS] min_samples_leaf = 5")
    print("  [PASS] subsample = 1.0")
    print("  [PASS] random_state = 42")
    print("  [PASS] training_protocol = expanding_refit_each_target")
    print("  [PASS] exact 7 return-history features")
    print("  [PASS] market features = false")
    print()
    print("Canonical-source guard")
    print("  [PASS] final_evaluation.yaml and model_freeze.yaml agree on G04")
    print("  [PASS] final_run_metadata.json records the same evaluated parameters")
    print("  [PASS] configs/gradient_boosting.yaml is detected as prototype G01 (lr=0.05), not canonical G04")
    print("  [PASS] runner injects configured G04 parameters; it does not rely on the class lr=0.05 default")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL - B28.2 FROZEN SPECIFICATION AUDIT: {exc}", file=sys.stderr)
        raise SystemExit(1)
