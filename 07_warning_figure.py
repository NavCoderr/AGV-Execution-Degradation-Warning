from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/pre_onset_experiments/"
            "09_pre_onset_predictions.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "figures/final_successful_h5_warning.png"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {args.predictions}"
        )

    data = pd.read_csv(args.predictions)

    required_columns = {
        "session",
        "timestamp",
        "mission_leg_id",
        "horizon_s",
        "variant",
        "feature_set",
        "model",
        "probability",
        "threshold",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="raise",
    )

    session = "S4_SAFETY_RICH_NAVEEN12"

    mission_leg = (
        "S4_SAFETY_RICH_NAVEEN12_L59"
    )

    onset_time = pd.Timestamp(
        "2026-03-01 01:40:06"
    )

    case = data[
        (data["session"] == session)
        & (
            data["mission_leg_id"]
            == mission_leg
        )
        & (data["horizon_s"] == 5)
        & (
            data["variant"]
            == "all_sessions"
        )
        & (
            data["feature_set"]
            == "full_graph_no_soc"
        )
        & (
            data["model"]
            == "RandomForest"
        )
        & (
            data["timestamp"]
            >= onset_time
            - pd.Timedelta(seconds=15)
        )
        & (
            data["timestamp"]
            < onset_time
        )
    ].copy()

    if case.empty:
        raise RuntimeError(
            "The selected held-out warning "
            "case was not found."
        )

    case = case.sort_values(
        "timestamp"
    )

    case["time_relative_to_onset_s"] = (
        case["timestamp"]
        - onset_time
    ).dt.total_seconds()

    threshold_values = (
        case["threshold"]
        .dropna()
        .unique()
    )

    if len(threshold_values) != 1:
        raise RuntimeError(
            "The selected case does not "
            "contain one consistent threshold."
        )

    threshold = float(
        threshold_values[0]
    )

    case["above_threshold"] = (
        case["probability"]
        >= threshold
    )

    case["alert_active"] = (
        case["above_threshold"]
        .astype(int)
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
        >= 2
    )

    alert_rows = case[
        case["alert_active"]
    ]

    if alert_rows.empty:
        raise RuntimeError(
            "The selected case does not "
            "activate the two-of-three "
            "alert policy."
        )

    alert_row = alert_rows.iloc[0]

    alert_time = float(
        alert_row[
            "time_relative_to_onset_s"
        ]
    )

    alert_probability = float(
        alert_row["probability"]
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 4.2)
    )

    axis.plot(
        case[
            "time_relative_to_onset_s"
        ],
        case["probability"],
        marker="o",
        linewidth=2,
        label=(
            "Calibrated onset probability"
        ),
    )

    axis.axhline(
        threshold,
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Fold threshold = "
            f"{threshold:.2f}"
        ),
    )

    axis.axvspan(
        -5,
        0,
        alpha=0.18,
        label=(
            "Onset within the next 5 s"
        ),
    )

    axis.axvline(
        0,
        linestyle=":",
        linewidth=1.8,
        label=(
            "Operational onset"
        ),
    )

    axis.scatter(
        [alert_time],
        [alert_probability],
        s=90,
        marker="D",
        zorder=5,
        label=(
            f"Alert activates at "
            f"{abs(alert_time):.0f} s lead"
        ),
    )

    axis.set_xlabel(
        "Time relative to onset (s)"
    )

    axis.set_ylabel(
        "Calibrated onset probability"
    )

    axis.set_xlim(
        -15,
        0.5,
    )

    axis.set_ylim(
        0,
        0.75,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend(
        fontsize=8,
        loc="upper left",
    )

    figure.tight_layout()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved: {args.output}"
    )

    print(
        f"Alert lead: "
        f"{abs(alert_time):.0f} s; "
        f"threshold: {threshold:.2f}"
    )


if __name__ == "__main__":
    main()