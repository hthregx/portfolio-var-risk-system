from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = REPO_ROOT / "results" / "final_metric_comparison.csv"
FIGURE_DIR = REPO_ROOT / "results" / "figures"

METHOD_ORDER = [
    "historical_simulation",
    "ewma",
    "gradient_boosting",
]

METHOD_LABELS = {
    "historical_simulation": "Historical Simulation",
    "ewma": "EWMA",
    "gradient_boosting": "Gradient Boosting",
}

OUTPUT_PATHS = {
    "violation_rate": FIGURE_DIR / "final_violation_rate_a.png",
    "pinball_loss": FIGURE_DIR / "final_pinball_loss_a.png",
    "average_var": FIGURE_DIR / "final_average_var_a.png",
}

REQUIRED_COLUMNS = {
    "method",
    "violation_rate",
    "nominal_violation_rate",
    "calibration_distance",
    "pinball_loss",
    "average_var",
    "calibration_leader",
    "pinball_leader",
    "lowest_average_var",
}


def load_frozen_comparison() -> pd.DataFrame:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen comparison artifact: {INPUT_PATH}")

    frame = pd.read_csv(INPUT_PATH)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            "Frozen comparison artifact is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(frame) != 3:
        raise ValueError(f"Expected 3 methods, found {len(frame)}")

    methods = set(frame["method"])
    expected_methods = set(METHOD_ORDER)

    if methods != expected_methods:
        raise ValueError(
            f"Unexpected method set: {sorted(methods)}; "
            f"expected {sorted(expected_methods)}"
        )

    if frame["method"].duplicated().any():
        raise ValueError("Duplicate method rows found in frozen comparison artifact")

    numeric_columns = [
        "violation_rate",
        "nominal_violation_rate",
        "calibration_distance",
        "pinball_loss",
        "average_var",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="raise")

        if values.isna().any():
            raise ValueError(f"Missing numeric values in {column}")

        frame[column] = values.astype(float)

    frame = frame.set_index("method").loc[METHOD_ORDER].reset_index()

    nominal_rates = frame["nominal_violation_rate"].unique()

    if len(nominal_rates) != 1:
        raise ValueError("Expected one shared nominal violation rate")

    nominal_rate = float(nominal_rates[0])

    expected_distance = (frame["violation_rate"] - nominal_rate).abs()
    observed_distance = frame["calibration_distance"]

    if not expected_distance.equals(observed_distance):
        max_difference = float((expected_distance - observed_distance).abs().max())

        if max_difference > 1e-12:
            raise ValueError(
                "Calibration distance is inconsistent with violation rate; "
                f"max difference={max_difference}"
            )

    calibration_leader = frame.loc[
        frame["calibration_distance"].idxmin(), "method"
    ]
    pinball_leader = frame.loc[frame["pinball_loss"].idxmin(), "method"]
    lowest_average_var = frame.loc[frame["average_var"].idxmin(), "method"]

    if calibration_leader != "ewma":
        raise ValueError(f"Unexpected calibration leader: {calibration_leader}")

    if pinball_leader != "gradient_boosting":
        raise ValueError(f"Unexpected pinball leader: {pinball_leader}")

    if lowest_average_var != "historical_simulation":
        raise ValueError(
            f"Unexpected lowest-average-VaR method: {lowest_average_var}"
        )

    return frame


def method_labels(frame: pd.DataFrame) -> list[str]:
    return [METHOD_LABELS[method] for method in frame["method"]]


def save_violation_rate_figure(frame: pd.DataFrame) -> None:
    nominal_rate = float(frame["nominal_violation_rate"].iloc[0])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    values = frame["violation_rate"] * 100.0
    labels = method_labels(frame)

    bars = ax.bar(labels, values)

    ax.axhline(
        nominal_rate * 100.0,
        linestyle="--",
        linewidth=1.2,
        label=f"Nominal tail probability ({nominal_rate:.0%})",
    )

    ax.set_title("Frozen Evaluation: Violation Rate")
    ax.set_ylabel("Violation rate (%)")
    ax.set_xlabel("Method")
    ax.legend()

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATHS["violation_rate"],
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_pinball_loss_figure(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    values = frame["pinball_loss"]
    labels = method_labels(frame)

    bars = ax.bar(labels, values)

    ax.set_title("Frozen Evaluation: Pinball Loss")
    ax.set_ylabel("Mean pinball loss")
    ax.set_xlabel("Method")

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.6f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATHS["pinball_loss"],
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_average_var_figure(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    values = frame["average_var"] * 100.0
    labels = method_labels(frame)

    bars = ax.bar(labels, values)

    ax.set_title("Frozen Evaluation: Average VaR")
    ax.set_ylabel("Average VaR (%)")
    ax.set_xlabel("Method")

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    fig.text(
        0.5,
        0.01,
        "Lower average VaR does not imply superior calibration "
        "or overall model superiority.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(
        OUTPUT_PATHS["average_var"],
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    frame = load_frozen_comparison()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_violation_rate_figure(frame)
    save_pinball_loss_figure(frame)
    save_average_var_figure(frame)

    print("Frozen evidence figures generated:")
    for path in OUTPUT_PATHS.values():
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()