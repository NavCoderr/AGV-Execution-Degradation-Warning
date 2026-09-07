from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


MOTION_HISTORY_FEATURES = {
    "speed_mps",
    "speed_mean_5s",
    "speed_mean_10s",
    "speed_mean_30s",
    "speed_std_10s",
    "stop_share_5s",
    "stop_share_10s",
    "stop_share_30s",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("agv_scientific_experiments", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "outputs" / "harmonized_graph_mission_state.csv"
    base_script = script_dir / "02_run_scientific_experiments.py"
    out_dir = script_dir / "outputs" / "target_feature_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\nRun 01_build_scientific_dataset.py first."
        )
    if not base_script.exists():
        raise FileNotFoundError(f"Base experiment script not found: {base_script}")

    exp = load_module(base_script)
    data = pd.read_csv(data_file, parse_dates=["timestamp"], low_memory=False)

    full_graph_features = exp.BASE + exp.GRAPH + exp.EUCLID
    euclidean_features = exp.BASE + exp.EUCLID

    feature_sets = {
        "euclidean_no_soc_motion_reduced": [
            c for c in euclidean_features
            if c not in exp.SOC_FEATURES and c not in MOTION_HISTORY_FEATURES
        ],
        "full_graph_no_soc_motion_reduced": [
            c for c in full_graph_features
            if c not in exp.SOC_FEATURES and c not in MOTION_HISTORY_FEATURES
        ],
    }

    all_models = exp.model_dict(False)
    rf = all_models["RandomForest"]

    rows: list[dict] = []

    for horizon in (5, 10):
        label = f"label_h{horizon}"
        eligible = data[
            data["at_risk"].eq(1)
            & data["active_mission"].eq(1)
            & data["external_hold"].eq(0)
            & data["telemetry_fresh"].eq(1)
            & data[label].notna()
        ].copy()

        for feature_name, requested_features in feature_sets.items():
            requested_num = [c for c in requested_features if c in eligible.columns]
            requested_cat = [c for c in exp.CAT if c in eligible.columns]

            for test_session in sorted(eligible["session"].dropna().unique()):
                test = eligible[eligible["session"] == test_session].copy()
                train = eligible[eligible["session"] != test_session].copy()

                if len(test) < 20 or train[label].nunique() < 2:
                    continue

                try:
                    fit, calibration, calibration_source = exp.split_fit_calibration(train, label)
                except RuntimeError:
                    continue
                if fit[label].nunique() < 2 or calibration[label].nunique() < 2:
                    continue

                num, cat, removed_all_missing = exp.usable_features_from_fit(
                    fit, requested_num, requested_cat
                )
                if not num and not cat:
                    continue

                pipe = Pipeline([
                    ("prep", exp.preprocessor(num, cat)),
                    ("model", rf),
                ])
                pipe.fit(fit[num + cat], fit[label].astype(int))

                p_cal_raw = pipe.predict_proba(calibration[num + cat])[:, 1]
                calibrator, calibration_method = exp.fit_platt_calibrator(
                    calibration[label].astype(int).to_numpy(), p_cal_raw
                )
                p_cal = exp.apply_probability_calibrator(calibrator, p_cal_raw)
                threshold = exp.select_threshold(
                    calibration[label].astype(int).to_numpy(), p_cal
                )

                p_test_raw = pipe.predict_proba(test[num + cat])[:, 1]
                p_test = exp.apply_probability_calibrator(calibrator, p_test_raw)
                y_test = test[label].astype(int).to_numpy()
                metrics = exp.score_binary(y_test, p_test, threshold)
                metrics.update({
                    "horizon_s": horizon,
                    "feature_set": feature_name,
                    "model": "RandomForest",
                    "test_session": test_session,
                    "calibration_source": calibration_source,
                    "calibration_method": calibration_method,
                    "n_used_features": len(num) + len(cat),
                    "removed_all_missing_features": "|".join(removed_all_missing),
                    "removed_motion_features": "|".join(sorted(MOTION_HISTORY_FEATURES)),
                })
                rows.append(metrics)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No sensitivity folds were produced.")

    results.to_csv(out_dir / "01_loso_results.csv", index=False)

    summary_rows = []
    for (horizon, feature_set), group in results.groupby(["horizon_s", "feature_set"]):
        pr = pd.to_numeric(group["pr_auc"], errors="coerce")
        summary_rows.append({
            "horizon_s": int(horizon),
            "feature_set": feature_set,
            "positive_event_folds": int(pr.notna().sum()),
            "pr_auc_mean": float(pr.mean()),
            "pr_auc_sd": float(pr.std(ddof=1)),
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "recall_mean": float(group["recall"].mean()),
            "f1_positive_mean": float(group["f1_positive"].mean()),
            "brier_mean": float(group["brier"].mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["horizon_s", "feature_set"])
    summary.to_csv(out_dir / "02_summary.csv", index=False)

    # Compact paper/supplement figure: matched Euclidean vs graph after removing
    # direct speed/stop-history predictors. Error bars are SD across positive-event folds.
    labels = ["5 s", "10 s"]
    euclid = []
    euclid_sd = []
    graph = []
    graph_sd = []
    for h in (5, 10):
        e = summary[(summary["horizon_s"] == h) &
                    (summary["feature_set"] == "euclidean_no_soc_motion_reduced")].iloc[0]
        g = summary[(summary["horizon_s"] == h) &
                    (summary["feature_set"] == "full_graph_no_soc_motion_reduced")].iloc[0]
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
    ax.set_title("Motion-history-reduced sensitivity")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "03_motion_history_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Target-feature sensitivity completed.")
    print(summary.to_string(index=False))
    print(f"Results: {out_dir / '01_loso_results.csv'}")
    print(f"Summary: {out_dir / '02_summary.csv'}")
    print(f"Figure: {out_dir / '03_motion_history_sensitivity.png'}")


if __name__ == "__main__":
    main()
