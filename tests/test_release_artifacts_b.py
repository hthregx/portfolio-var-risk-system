from scripts.audit_release_artifacts_b import (
    EXPECTED_DATES,
    EXPECTED_ROWS,
    audit_contract,
    audit_expected_metrics,
    audit_semantics,
    load_predictions,
    recompute_metrics,
)


def test_release_contract():
    df = load_predictions()

    audit_contract(df)

    assert len(df) == EXPECTED_ROWS
    assert df["target_date"].nunique() == EXPECTED_DATES


def test_release_semantics():
    df = load_predictions()

    audit_semantics(df)


def test_release_metrics_match_expected():
    df = load_predictions()

    metrics = recompute_metrics(df)

    audit_expected_metrics(metrics)


def test_all_methods_have_398_rows():
    df = load_predictions()

    counts = df["method"].value_counts()

    assert (counts == 398).all()


def test_no_missing_values():
    df = load_predictions()

    assert not df.isna().any().any()