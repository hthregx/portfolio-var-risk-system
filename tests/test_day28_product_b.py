from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "app.py"


def load_source() -> str:
    assert APP_PATH.is_file(), "missing app/app.py"
    return APP_PATH.read_text(encoding="utf-8")


def load_tree() -> ast.Module:
    return ast.parse(load_source(), filename=str(APP_PATH))


def function_node(name: str) -> ast.FunctionDef:
    for node in load_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function: {name}")


def source_of_function(name: str) -> str:
    source = load_source()
    node = function_node(name)
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _is_value_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "value"


def _is_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and node.value == 0
    )


def _condition_rejects_zero(test: ast.AST) -> bool:
    """
    Accept common positive-value guards:
      value <= 0
      value == 0
      0 >= value
      0 == value
      not value > 0
    Negative values are already independently rejected by parse_portfolio_value.
    """
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
        left = test.left
        right = test.comparators[0]
        op = test.ops[0]

        if _is_value_name(left) and _is_zero(right):
            return isinstance(op, (ast.LtE, ast.Eq))
        if _is_zero(left) and _is_value_name(right):
            return isinstance(op, (ast.GtE, ast.Eq))

    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        inner = test.operand
        if (
            isinstance(inner, ast.Compare)
            and len(inner.ops) == 1
            and len(inner.comparators) == 1
            and _is_value_name(inner.left)
            and _is_zero(inner.comparators[0])
            and isinstance(inner.ops[0], ast.Gt)
        ):
            return True

    return False


def _if_rejects_with_none(node: ast.If) -> bool:
    if not _condition_rejects_zero(node.test):
        return False

    has_error = any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and isinstance(child.func.value, ast.Name)
        and child.func.value.id == "st"
        and child.func.attr == "error"
        for statement in node.body
        for child in ast.walk(statement)
    )
    has_none_return = any(
        isinstance(statement, ast.Return) and statement.value is None
        for statement in node.body
    )
    return has_error and has_none_return


def test_product_contract_portfolio_and_horizon() -> None:
    source = load_source()

    assert "HPG / FPT / MWG" in source
    assert "equal-weight" in source.lower()
    assert "one-day 95% var" in source.lower()
    assert "one trading session" in source.lower()


def test_product_contract_exact_three_model_labels() -> None:
    tree = load_tree()

    labels = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "MODEL_LABELS" for target in node.targets):
                labels = ast.literal_eval(node.value)
                break

    assert labels == {
        "historical_simulation": "Historical Simulation",
        "ewma": "EWMA",
        "gradient_boosting": "Gradient Boosting G04",
    }


def test_product_contract_frozen_and_operational_dates_are_separate() -> None:
    source = load_source()

    assert "2026-07-28" in source
    assert "2026-08-28" in source
    assert "Canonical research cutoff" in source
    assert "Operational data cutoff" in source
    assert "Operational refreshes do not rewrite canonical evaluation results." in source


def test_product_contract_interpretation_guardrails() -> None:
    """
    Read user-visible string literals from the AST so adjacent Python string
    literals split across source lines are evaluated the same way Python
    evaluates them.
    """
    tree = load_tree()
    visible_text = " ".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ).lower()

    assert "does not define a single overall winning model" in visible_text
    assert "not the maximum possible loss" in visible_text
    assert "descriptive, not an accuracy ranking" in visible_text


def test_product_contract_vnd_scaling_is_direct_var_scaling() -> None:
    source = load_source()

    assert '"Portfolio value (VND)"' in source
    assert "estimated_loss = portfolio_value * var_return" in source
    assert '"Estimated VaR amount"' in source


def test_product_contract_portfolio_value_must_be_strictly_positive() -> None:
    """
    Product acceptance requires portfolio value > 0.
    B is read-only on app.py; if this test fails, A must add a zero-value guard.
    """
    node = function_node("parse_portfolio_value")

    guards = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.If) and _if_rejects_with_none(child)
    ]

    assert guards, (
        "parse_portfolio_value currently permits 0 VND. "
        "Product acceptance requires portfolio value > 0. "
        "A-owned app.py should reject value <= 0 (or explicitly value == 0) "
        "with st.error(...) and return None."
    )


def test_product_contract_input_rejects_negative_and_invalid_text() -> None:
    source = source_of_function("parse_portfolio_value")

    assert 'raw.startswith("-")' in source
    assert "Portfolio value cannot be negative." in source
    assert 're.fullmatch(r"[0-9\\s.,_]+", raw)' in source


def test_product_contract_dashboard_does_not_retrain_models() -> None:
    source = load_source().lower()

    forbidden = (
        "gradientboostingregressor(",
        ".fit(",
        "run_final_walk_forward(",
    )
    for token in forbidden:
        assert token not in source
