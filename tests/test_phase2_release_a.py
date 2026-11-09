import numpy as np
import pandas as pd
import pytest

from src.phase2.release_a import validate_holdout, validate_offline_html


def fixture():
    dates = pd.bdate_range("2024-01-01", periods=468)
    d = dates.strftime("%Y-%m-%d").tolist()
    previous = (dates - pd.offsets.BDay(1)).strftime("%Y-%m-%d").tolist()
    portfolio = pd.DataFrame({"date": d, "portfolio_simple_return": np.linspace(-0.02, 0.02, 468)})
    q = float(portfolio["portfolio_simple_return"].iloc[25])
    out = []
    for model in ("gb_G04_7f_a", "gbv2_12f_c1_a"):
        out.append(pd.DataFrame({
            "model": model,
            "as_of": previous,
            "target_date": d,
            "quantile_return": q,
            "var_return": max(0.0, -q),
            "actual_return": portfolio["portfolio_simple_return"].to_numpy(),
        }))
    pred = pd.concat(out, ignore_index=True)
    return pred, portfolio


def test_accept_valid_matched_holdout():
    pred, portfolio = fixture()
    metrics = validate_holdout(pred, portfolio)
    assert metrics["gb_G04_7f_a"]["exceptions"] == 25
    assert metrics["gbv2_12f_c1_a"]["exceptions"] == 25


def test_reject_duplicate_target():
    p, r = fixture()
    p.loc[1, "target_date"] = p.loc[0, "target_date"]
    with pytest.raises(ValueError, match="duplicate target"):
        validate_holdout(p, r)


def test_reject_invalid_var_sign():
    p, r = fixture()
    p.loc[0, "var_return"] = -0.015
    with pytest.raises(ValueError, match="VaR sign"):
        validate_holdout(p, r)


def test_reject_lookahead():
    p, r = fixture()
    p.loc[0, "as_of"] = p.loc[0, "target_date"]
    with pytest.raises(ValueError, match="leaked"):
        validate_holdout(p, r)


def test_reject_false_actual_return():
    p, r = fixture()
    p.loc[0, "actual_return"] = 0.3
    with pytest.raises(ValueError, match="actual return mismatch"):
        validate_holdout(p, r)


def test_reject_different_model_dates():
    p, r = fixture()
    next_day = (pd.Timestamp(r["date"].iloc[-1]) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    r = pd.concat([r, pd.DataFrame({"date": [next_day], "portfolio_simple_return": [0.02]})], ignore_index=True)
    p.loc[935, "as_of"] = p.loc[935, "target_date"]
    p.loc[935, "target_date"] = next_day
    p.loc[935, "actual_return"] = 0.02
    with pytest.raises(ValueError, match="not directly comparable"):
        validate_holdout(p, r)


def test_offline_html_accepts_embedded_images(tmp_path):
    path = tmp_path / "index.html"
    path.write_text("<html>GBv2 " + "x" * 1100 + "<img src='data:image/png;base64,AA=='></html>", encoding="utf-8")
    assert validate_offline_html(path) == 1


def test_offline_html_rejects_broken_image(tmp_path):
    path = tmp_path / "index.html"
    path.write_text("<html>GBv2 " + "x" * 1100 + "<img src='missing.png'></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="broken image"):
        validate_offline_html(path)


def test_offline_html_rejects_external_image(tmp_path):
    path = tmp_path / "index.html"
    path.write_text("<html>GBv2 " + "x" * 1100 + "<img src='https://example.com/a.png'></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="external image"):
        validate_offline_html(path)
