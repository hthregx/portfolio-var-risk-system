from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase2.result_adapter_a import matched_holdout, mean_pinball, normalize_predictions
from scripts.run_phase2_integration_a import render_html


def sample(model="gb_G04_7f_a", start="2026-01-02"):
    d = pd.bdate_range(start, periods=3)
    return pd.DataFrame({
        "model": [model]*3, "fold": ["holdout_fixed_fit"]*3,
        "as_of": ["2026-01-01", "2026-01-02", "2026-01-05"],
        "target_date": d,
        "quantile_return": [-.02]*3, "var_return": [.02]*3,
        "actual_return": [-.03, .01, -.02],
        "violation": [True, False, False],
    })


def test_normalize_preserves_dates_and_sign():
    df = normalize_predictions(sample(), source="A", protocol="fixed_fit_retrospective_holdout")
    assert len(df) == 3 and df.source.unique().tolist() == ["A"]
    assert df.violation.tolist() == [True, False, False]
    assert df.target_date.iloc[0] == pd.Timestamp("2026-01-02")
    assert df.var_return.iloc[0] == pytest.approx(.02)


def test_target_equality_not_violation():
    df = normalize_predictions(sample(), source="B", protocol="development_folds")
    assert not bool(df.violation.iloc[2])


def test_wrong_var_rejected():
    x = sample()
    x.loc[0, "var_return"] = -.02
    with pytest.raises(ValueError, match="magnitude/sign"):
        normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")


def test_duplicate_target_rejected():
    x = sample()
    x.loc[1, "target_date"] = x.loc[0, "target_date"]
    x.loc[1, "as_of"] = "2026-01-01"
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")


def test_asof_must_precede():
    x = sample()
    x.loc[0, "as_of"] = x.loc[0, "target_date"]
    with pytest.raises(ValueError, match="predate"):
        normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")


def test_wrong_violation_rejected():
    x = sample()
    x.loc[0, "violation"] = False
    with pytest.raises(ValueError, match="exceptions"):
        normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")


def test_nonfinite_rejected():
    x = sample()
    x.loc[0, "quantile_return"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Missing"):
        normalize_predictions(sample().drop(columns="as_of"), source="A", protocol="x")


def test_b_cannot_be_compared_in_a_holdout():
    a = normalize_predictions(sample(), source="B", protocol="fixed_fit_retrospective_holdout")
    with pytest.raises(ValueError, match="Only A"):
        matched_holdout(a, "gb_G04_7f_a", "gbv2_12f_c1_a")


def test_different_targets_disallowed():
    x = pd.concat([sample(), sample("gbv2_12f_c1_a", start="2026-01-05")], ignore_index=True)
    n = normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")
    with pytest.raises(ValueError, match="Unequal forecast target"):
        matched_holdout(n, "gb_G04_7f_a", "gbv2_12f_c1_a")


def test_different_observations_disallowed():
    x = pd.concat([sample(), sample("gbv2_12f_c1_a")], ignore_index=True)
    x.loc[3, "actual_return"] = -.04
    n = normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")
    with pytest.raises(ValueError, match="Unequal observed"):
        matched_holdout(n, "gb_G04_7f_a", "gbv2_12f_c1_a")


def test_match_works_on_same_dates():
    x = pd.concat([sample(), sample("gbv2_12f_c1_a")], ignore_index=True)
    n = normalize_predictions(x, source="A", protocol="fixed_fit_retrospective_holdout")
    a,b = matched_holdout(n,"gb_G04_7f_a","gbv2_12f_c1_a")
    assert len(a) == len(b) == 3


def test_pinball_hand_calculation():
    assert mean_pinball([-0.02, 0.02], [-0.01, -0.01]) == pytest.approx((.0095+.0015)/2)


def test_html_is_not_live_or_overclaiming():
    ranking = pd.DataFrame({"model":["gbv2_12f_c1_a"], "cv_pinball":[.002], "n_features":[12]})
    hold = pd.DataFrame({"model":["gbv2_12f_c1_a"],"n":[468], "violations":[25], "violation_rate":[25/468], "pinball_loss":[.0016], "average_var":[.023]})
    back = pd.DataFrame({"model":["gbv2_12f_c1_a"],"n":[468],"violations":[25],"violation_rate":[25/468],"p_uc":[.7],"p_ind":[.5],"p_cc":[.8]})
    b = pd.DataFrame({"model":["Historical Simulation w250"],"prediction_rows":[1455],"exception_count":[76],"violation_rate":[.05],"mean_pinball_loss":[.002]})
    page=render_html(ranking=ranking,hold=hold,back=back,b=b,candidate="gbv2_12f_c1_a",percentage=6.5,picture_files=["figures/actual_vs_quantile.png"])
    assert "không phải dự báo thời gian thực" in page
    assert "trước 18/12/2024" in page
    assert "không được so sánh" in page
    assert "GBv2 C1" in page


def test_a4_manifest_accepts_windows_style_paths_and_detects_tampering(tmp_path):
    import hashlib
    import json
    from scripts.run_phase2_integration_a import verify_a4

    train = tmp_path / "training"
    train.mkdir()
    (train / "metadata.json").write_text("{}", encoding="utf-8")
    folder = tmp_path / "backtesting"
    folder.mkdir()

    needed = [
        "holdout_backtests.csv", "cv_fold_metrics_verified.csv",
        "holdout_paired_comparison.csv", "holdout_exceptions.csv",
        "figures/actual_vs_quantile.png", "figures/cumulative_exceptions.png",
        "figures/cv_fold_pinball.png", "figures/holdout_pinball_comparison.png",
        "figures/rolling_60d_violation_rate.png",
    ]
    hashes = {}
    for rel in needed:
        f = folder / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rel.encode("utf-8"))
        hashes[rel.replace("/", "\\")] = hashlib.sha256(f.read_bytes()).hexdigest()

    selected = "gbv2_12f_c1_a"
    manifest = {
        "source_training_metadata_sha256": hashlib.sha256(
            (train / "metadata.json").read_bytes()
        ).hexdigest(),
        "source_data_sha256": "test-source-hash",
        "selected_v2_on_cv": selected,
        "do_not_use_for_selection": True,
        "files": hashes,
    }
    (folder / "a4_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    meta = {"source_sha256": "test-source-hash", "selected_v2_on_cv": selected}
    assert verify_a4(folder, train, meta) == manifest

    (folder / "figures/actual_vs_quantile.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_a4(folder, train, meta)
