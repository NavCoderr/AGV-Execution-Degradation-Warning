from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    prediction_file = script_dir / "outputs" / "pre_onset_experiments" / "09_pre_onset_predictions.csv"
    data_file = script_dir / "outputs" / "harmonized_graph_mission_state.csv"
    out_dir = script_dir / "outputs" / "pre_candidate_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Predictions not found: {prediction_file}\n"
            "Run 03_run_pre_onset_experiments.py first."
        )
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\nRun 01_build_scientific_dataset.py first."
        )

    predictions = pd.read_csv(prediction_file, parse_dates=["timestamp"], low_memory=False)
    state = pd.read_csv(
        data_file,
        usecols=["session", "timestamp", "degradation_candidate"],
        parse_dates=["timestamp"],
        low_memory=False,
    )

    predictions = predictions[
        predictions["variant"].eq("all_sessions")
        & predictions["model"].eq("RandomForest")
        & predictions["feature_set"].isin(["euclidean_no_soc", "full_graph_no_soc"])
    ].copy()

    predictions = predictions.merge(
        state,
        on=["session", "timestamp"],
        how="left",
        validate="many_to_one",
    )
    predictions = predictions[predictions["degradation_candidate"].eq(0)].copy()

    fold_rows = []
    for horizon in (5, 10, 20):
        label = f"label_h{horizon}"
        for feature_set in ("euclidean_no_soc", "full_graph_no_soc"):
            subset = predictions[
                predictions["horizon_s"].eq(horizon)
                & predictions["feature_set"].eq(feature_set)
            ].copy()
            for session, group in subset.groupby("session"):
                y = group[label].astype(int).to_numpy()
                if len(np.unique(y)) < 2:
                    pr_auc = np.nan
                else:
                    pr_auc = float(average_precision_score(y, group["probability"].to_numpy()))
                fold_rows.append({
                    "horizon_s": horizon,
                    "feature_set": feature_set,
                    "test_session": session,
                    "rows": int(len(group)),
                    "positive_anchors": int(y.sum()),
                    "pr_auc": pr_auc,
                })

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_dir / "01_fold_results.csv", index=False)

    summary_rows = []
    for (horizon, feature_set), group in folds.groupby(["horizon_s", "feature_set"]):
        values = pd.to_numeric(group["pr_auc"], errors="coerce")
        summary_rows.append({
            "horizon_s": int(horizon),
            "feature_set": feature_set,
            "positive_event_folds": int(values.notna().sum()),
            "pr_auc_mean": float(values.mean()),
            "pr_auc_sd": float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan,
            "rows_total": int(group["rows"].sum()),
            "positive_anchors_total": int(group["positive_anchors"].sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["horizon_s", "feature_set"])
    summary.to_csv(out_dir / "02_summary.csv", index=False)

    labels = ["5 s", "10 s", "20 s"]
    euclid, euclid_sd, graph, graph_sd = [], [], [], []
    for h in (5, 10, 20):
        e = summary[(summary["horizon_s"] == h) & (summary["feature_set"] == "euclidean_no_soc")].iloc[0]
        g = summary[(summary["horizon_s"] == h) & (summary["feature_set"] == "full_graph_no_soc")].iloc[0]
        euclid.append(e["pr_auc_mean"])
        euclid_sd.append(e["pr_auc_sd"])
        graph.append(g["pr_auc_mean"])
        graph_sd.append(g["pr_auc_sd"])

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(x - width / 2, euclid, width, yerr=euclid_sd, capsize=4,
           label="Euclidean, no SOC")
    ax.bar(x + width / 2, graph, width, yerr=graph_sd, capsize=4,
           label="Euclidean + graph, no SOC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean PR-AUC")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Candidate-free anchor sensitivity")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "03_pre_candidate_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Pre-candidate sensitivity completed.")
    print(summary.to_string(index=False))
    print(f"Fold results: {out_dir / '01_fold_results.csv'}")
    print(f"Summary: {out_dir / '02_summary.csv'}")
    print(f"Figure: {out_dir / '03_pre_candidate_sensitivity.png'}")


if __name__ == "__main__":
    main()
