

from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_pipeline_module(path: Path):
    spec = importlib.util.spec_from_file_location("agv_final_experiments", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aggregate_summary(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy", "macro_f1", "precision", "recall", "f1_positive",
        "roc_auc", "pr_auc", "brier", "positive_rate",
    ]
    rows = []
    group_cols = ["horizon_s", "variant", "feature_set", "model"]
    for keys, group in results.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["folds"] = int(group["test_session"].nunique())
        row["test_rows_total"] = int(group["n"].sum())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_evaluable_folds"] = int(values.notna().sum())
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "outputs" / "harmonized_graph_mission_state.csv"
    base_script = script_dir / "02_run_scientific_experiments.py"
    out_dir = script_dir / "outputs" / "pre_onset_experiments"
    fast = False

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            "Run 01_build_scientific_dataset.py first."
        )
    if not base_script.exists():
        raise FileNotFoundError(
            f"Base experiment script not found: {base_script}"
        )

    exp = load_pipeline_module(base_script)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_file, parse_dates=["timestamp"], low_memory=False)
    required = {
        "session", "timestamp", "active_mission", "external_hold",
        "telemetry_fresh", "current_degraded", "current_state_valid",
        "at_risk", "time_to_onset_s", "source_annotation_quality",
        "route_source", "temporal_segment_id", "mission_leg_id",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    variants = {
        "all_sessions": pd.Series(True, index=data.index),
        "annotated_sessions": data["source_annotation_quality"].ne("telemetry_only"),
        "recorded_route_only": data["route_source"].eq("recorded"),
    }

    full_graph_features = exp.BASE + exp.GRAPH + exp.EUCLID
    feature_sets = {
        "no_graph": exp.BASE,
        "euclidean_only": exp.BASE + exp.EUCLID,
        "full_graph": full_graph_features,
        "full_graph_no_soc": [
            c for c in full_graph_features if c not in exp.SOC_FEATURES
        ],
    }

    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon in (5, 10, 20):
        label = f"label_h{horizon}"
        if label not in data.columns:
            continue

        for variant_name, variant_mask in variants.items():
            # Primary strict pre-onset condition: a causally valid at-risk row.
            eligible = data[
                variant_mask
                & data["at_risk"].eq(1)
                & data["active_mission"].eq(1)
                & data["external_hold"].eq(0)
                & data["telemetry_fresh"].eq(1)
                & data[label].notna()
            ].copy()

            for feature_name, requested_features in feature_sets.items():
                # Keep sensitivity variants compact: full graph only.
                if variant_name != "all_sessions" and feature_name != "full_graph":
                    continue

                requested_num = [c for c in requested_features if c in eligible.columns]
                requested_cat = [c for c in exp.CAT if c in eligible.columns]

                for test_session in sorted(eligible["session"].dropna().unique()):
                    test = eligible[eligible["session"] == test_session].copy()
                    train = eligible[eligible["session"] != test_session].copy()

                    # A fold must have enough rows and both classes in development data.
                    if len(test) < 20 or train[label].nunique() < 2:
                        continue

                    try:
                        fit, calibration, calibration_source = exp.split_fit_calibration(train, label)
                    except RuntimeError:
                        continue
                    if fit[label].nunique() < 2 or calibration[label].nunique() < 2:
                        continue

                    num, cat, removed_features = exp.usable_features_from_fit(
                        fit, requested_num, requested_cat
                    )
                    if not num and not cat:
                        continue

                    for model_name, model in exp.model_dict(fast).items():
                        pipe = Pipeline([
                            ("prep", exp.preprocessor(num, cat)),
                            ("model", model),
                        ])

                        start = time.perf_counter()
                        pipe.fit(fit[num + cat], fit[label].astype(int))
                        fit_seconds = time.perf_counter() - start

                        p_cal_raw = pipe.predict_proba(calibration[num + cat])[:, 1]
                        calibrator, calibration_method = exp.fit_platt_calibrator(
                            calibration[label].astype(int).to_numpy(),
                            p_cal_raw,
                        )
                        p_cal = exp.apply_probability_calibrator(
                            calibrator,
                            p_cal_raw,
                        )
                        threshold = exp.select_threshold(
                            calibration[label].astype(int).to_numpy(), p_cal
                        )

                        start = time.perf_counter()
                        p_test_raw = pipe.predict_proba(test[num + cat])[:, 1]
                        p_test = exp.apply_probability_calibrator(
                            calibrator,
                            p_test_raw,
                        )
                        infer_ms_row = (
                            (time.perf_counter() - start) * 1000.0 / len(test)
                        )
                        y_test = test[label].astype(int).to_numpy()

                        metrics = exp.score_binary(y_test, p_test, threshold)
                        metrics.update({
                            "task": "pre_onset_currently_healthy",
                            "horizon_s": horizon,
                            "variant": variant_name,
                            "feature_set": feature_name,
                            "test_session": test_session,
                            "model": model_name,
                            "calibration_source": calibration_source,
                            "calibration_method": calibration_method,
                            "fit_rows": int(len(fit)),
                            "calibration_rows": int(len(calibration)),
                            "test_rows": int(len(test)),
                            "fit_s": float(fit_seconds),
                            "infer_ms_row": float(infer_ms_row),
                            "n_used_features": int(len(num) + len(cat)),
                            "removed_all_missing_features": "|".join(removed_features),
                        })
                        result_rows.append(metrics)

                        pred = test[[
                            "session", "timestamp", "temporal_segment_id",
                            "mission_leg_id", "current_degraded",
                            "time_to_onset_s", label,
                        ]].copy()
                        pred["task"] = "pre_onset_currently_healthy"
                        pred["horizon_s"] = horizon
                        pred["variant"] = variant_name
                        pred["feature_set"] = feature_name
                        pred["model"] = model_name
                        pred["probability"] = p_test
                        pred["threshold"] = threshold
                        pred["prediction"] = (p_test >= threshold).astype(int)
                        prediction_frames.append(pred)

                # End test-session loop.

        # Strict persistence baseline: current_degraded == 0 for every at-risk
        # row, so causal persistence is identical to AlwaysNegative.
        baseline_eligible = data[
            data["at_risk"].eq(1)
            & data["active_mission"].eq(1)
            & data["external_hold"].eq(0)
            & data["telemetry_fresh"].eq(1)
            & data[label].notna()
        ].copy()
        for test_session in sorted(baseline_eligible["session"].dropna().unique()):
            test = baseline_eligible[baseline_eligible["session"] == test_session]
            if len(test) < 20:
                continue
            y_test = test[label].astype(int).to_numpy()
            p_test = np.zeros(len(test), dtype=float)
            for baseline_name in ("AlwaysNegative", "PersistenceH0"):
                metrics = exp.score_binary(y_test, p_test, 0.5)
                metrics.update({
                    "task": "pre_onset_currently_healthy",
                    "horizon_s": horizon,
                    "variant": "all_sessions",
                    "feature_set": "engineering_baseline",
                    "test_session": test_session,
                    "model": baseline_name,
                    "calibration_source": "fixed_rule_no_training",
                    "calibration_method": "not_applicable",
                    "fit_rows": 0,
                    "calibration_rows": 0,
                    "test_rows": int(len(test)),
                    "fit_s": 0.0,
                    "infer_ms_row": 0.0,
                    "n_used_features": 0,
                    "removed_all_missing_features": "",
                })
                result_rows.append(metrics)

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError("No pre-onset folds were produced.")

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames else pd.DataFrame()
    )
    summary = aggregate_summary(results)

    results.to_csv(out_dir / "07_pre_onset_loso_results.csv", index=False)
    summary.to_csv(out_dir / "08_pre_onset_summary.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(out_dir / "09_pre_onset_predictions.csv", index=False)

    # best configurations per horizon by mean PR-AUC, with macro-F1
    # retained for context. Baselines are excluded from model selection.
    learned = summary[
        ~summary["model"].isin(["AlwaysNegative", "PersistenceH0"])
        & summary["variant"].eq("all_sessions")
    ]
    best = (
        learned.sort_values(
            ["horizon_s", "pr_auc_mean", "macro_f1_mean"],
            ascending=[True, False, False],
        )
        .groupby("horizon_s", as_index=False)
        .head(1)
    )
    best.to_csv(out_dir / "10_pre_onset_best_models.csv", index=False)

    print("Pre-onset experiment completed.")
    print(f"Results: {out_dir / '07_pre_onset_loso_results.csv'}")
    print(f"Summary: {out_dir / '08_pre_onset_summary.csv'}")
    print(f"Best models: {out_dir / '10_pre_onset_best_models.csv'}")
    print("\nBest learned configuration per horizon:")
    cols = [
        "horizon_s", "model", "feature_set", "variant", "folds",
        "macro_f1_mean", "macro_f1_sd", "pr_auc_mean", "pr_auc_sd",
        "recall_mean", "f1_positive_mean", "positive_rate_mean",
    ]
    print(best[cols].to_string(index=False))


if __name__ == "__main__":
    main()
