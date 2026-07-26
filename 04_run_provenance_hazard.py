from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
MAX_HORIZON = 20


def load_experiment_module(path: Path):
    spec = importlib.util.spec_from_file_location("agv_experiments", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment utilities: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_provenance_features(frame: pd.DataFrame, graph_features: list[str]) -> tuple[pd.DataFrame, list[str]]:

    out = frame.copy()
    recorded = out["route_source"].eq("recorded").astype(float)
    reconstructed = out["route_source"].eq("reconstructed_shortest").astype(float)
    available = (recorded + reconstructed).clip(0, 1)

    out["route_recorded"] = recorded
    out["route_reconstructed"] = reconstructed
    out["graph_context_available"] = available

    interaction_features = []
    for feature in graph_features:
        if feature not in out.columns:
            continue
        values = pd.to_numeric(out[feature], errors="coerce")
        recorded_name = f"recorded__{feature}"
        reconstructed_name = f"reconstructed__{feature}"
        out[recorded_name] = values * recorded
        out[reconstructed_name] = values * reconstructed
        interaction_features.extend([recorded_name, reconstructed_name])
    return out, interaction_features


def expand_person_period(
    anchors: pd.DataFrame,
    max_horizon: int = MAX_HORIZON,
) -> pd.DataFrame:
    """Expand at-risk anchors into discrete one-second hazard observations."""
    rows = []
    for anchor_id, row in anchors.iterrows():
        event_time = pd.to_numeric(
            pd.Series([row.get("time_to_onset_s", np.nan)]),
            errors="coerce",
        ).iloc[0]
        followup = pd.to_numeric(
            pd.Series([row.get("followup_available_s", 0)]),
            errors="coerce",
        ).iloc[0]
        event_time_int = (
            int(round(float(event_time)))
            if pd.notna(event_time) and 1 <= float(event_time) <= max_horizon
            else None
        )
        exposure = event_time_int if event_time_int is not None else int(
            min(max(float(followup), 0.0), max_horizon)
        )
        if exposure < 1:
            continue
        for hazard_bin in range(1, exposure + 1):
            expanded = row.to_dict()
            expanded["anchor_id"] = int(anchor_id)
            expanded["hazard_bin"] = str(hazard_bin)
            expanded["hazard_event"] = int(
                event_time_int is not None and hazard_bin == event_time_int
            )
            rows.append(expanded)
    return pd.DataFrame(rows)


def make_prediction_grid(
    anchors: pd.DataFrame,
    max_horizon: int = MAX_HORIZON,
) -> pd.DataFrame:
    rows = []
    for anchor_id, row in anchors.iterrows():
        for hazard_bin in range(1, max_horizon + 1):
            expanded = row.to_dict()
            expanded["anchor_id"] = int(anchor_id)
            expanded["hazard_bin"] = str(hazard_bin)
            rows.append(expanded)
    return pd.DataFrame(rows)


def cumulative_risk_from_grid(
    grid: pd.DataFrame,
    hazard_probability: np.ndarray,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    work = grid[["anchor_id", "hazard_bin"]].copy()
    work["hazard_bin"] = pd.to_numeric(work["hazard_bin"], errors="coerce").astype(int)
    work["hazard_probability"] = np.asarray(hazard_probability, dtype=float)
    work = work.sort_values(["anchor_id", "hazard_bin"])
    work["survival"] = (
        1.0 - work["hazard_probability"].clip(1e-7, 1 - 1e-7)
    ).groupby(work["anchor_id"]).cumprod()
    work["cumulative_risk"] = 1.0 - work["survival"]

    output = pd.DataFrame(index=sorted(work["anchor_id"].unique()))
    output.index.name = "anchor_id"
    for horizon in horizons:
        selected = work[work["hazard_bin"].eq(horizon)].set_index("anchor_id")
        output[f"risk_h{horizon}"] = selected["cumulative_risk"]
    return output


def coefficient_table(pipe: Pipeline, model_name: str) -> pd.DataFrame:
    try:
        names = pipe.named_steps["prep"].get_feature_names_out()
        coefficients = pipe.named_steps["model"].coef_[0]
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame({
        "model": model_name,
        "transformed_feature": names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "outputs" / "harmonized_graph_mission_state.csv"
    utilities_file = script_dir / "02_run_scientific_experiments.py"
    out_dir = script_dir / "outputs" / "provenance_hazard"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            "Run 01_build_scientific_dataset.py first."
        )
    exp = load_experiment_module(utilities_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_file, parse_dates=["timestamp"], low_memory=False)
    required = {
        "session", "timestamp", "temporal_segment_id", "mission_leg_id",
        "at_risk", "time_to_onset_s", "followup_available_s",
        "route_source", "label_h5", "label_h10", "label_h20",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    anchors = data[
        data["at_risk"].eq(1)
        & data["followup_available_s"].ge(1)
    ].copy()
    anchors = anchors.reset_index(drop=True)
    anchors.index.name = "anchor_id"
    anchors["split_event"] = anchors["time_to_onset_s"].between(1, MAX_HORIZON).astype(int)

    anchors, interaction_features = add_provenance_features(anchors, exp.GRAPH)
    quality_features = [
        "route_recorded", "route_reconstructed", "graph_context_available",
        "snap_distance_m", "position_confidence", "seconds_since_observed",
        "observed_sample", "short_causal_hold",
    ]

    model_specs = {
        "TelemetryHazard": {
            "num": exp.BASE,
            "cat": ["hazard_bin"],
        },
        "UngatedGraphHazard": {
            "num": exp.BASE + exp.EUCLID + exp.GRAPH,
            "cat": ["hazard_bin", "route_source"],
        },
        "ProvenanceConditionedHazard": {
            "num": exp.BASE + exp.EUCLID + quality_features + interaction_features,
            "cat": ["hazard_bin", "route_source"],
        },
    }

    result_rows = []
    prediction_rows = []
    coefficient_frames = []

    for test_session in sorted(anchors["session"].dropna().unique()):
        test = anchors[anchors["session"].eq(test_session)].copy()
        development = anchors[~anchors["session"].eq(test_session)].copy()
        if len(test) < 20 or development["split_event"].nunique() < 2:
            continue

        try:
            fit, calibration, calibration_source = exp.split_fit_calibration(
                development,
                "split_event",
            )
        except RuntimeError:
            continue

        fit_pp = expand_person_period(fit)
        if fit_pp.empty or fit_pp["hazard_event"].nunique() < 2:
            continue
        calibration_grid = make_prediction_grid(calibration)
        test_grid = make_prediction_grid(test)

        for model_name, spec in model_specs.items():
            requested_num = list(dict.fromkeys([
                column for column in spec["num"]
                if column in fit_pp.columns
            ]))
            requested_cat = list(dict.fromkeys([
                column for column in spec["cat"]
                if column in fit_pp.columns
            ]))
            num, cat, removed = exp.usable_features_from_fit(
                fit_pp,
                requested_num,
                requested_cat,
            )
            if not num and not cat:
                continue

            pipe = Pipeline([
                ("prep", exp.preprocessor(num, cat)),
                ("model", LogisticRegression(
                    C=0.25,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                )),
            ])
            pipe.fit(
                fit_pp[num + cat],
                fit_pp["hazard_event"].astype(int),
            )

            p_cal_hazard = pipe.predict_proba(
                calibration_grid[num + cat]
            )[:, 1]
            p_test_hazard = pipe.predict_proba(
                test_grid[num + cat]
            )[:, 1]
            calibration_risk = cumulative_risk_from_grid(
                calibration_grid,
                p_cal_hazard,
            )
            test_risk = cumulative_risk_from_grid(
                test_grid,
                p_test_hazard,
            )

            for horizon in (5, 10, 20):
                label = f"label_h{horizon}"
                cal_ids = calibration[
                    calibration[label].notna()
                ].index.intersection(calibration_risk.index)
                test_ids = test[
                    test[label].notna()
                ].index.intersection(test_risk.index)
                if len(cal_ids) < 20 or len(test_ids) < 20:
                    continue
                y_cal = calibration.loc[cal_ids, label].astype(int).to_numpy()
                y_test = test.loc[test_ids, label].astype(int).to_numpy()
                if len(np.unique(y_cal)) < 2:
                    continue

                raw_cal = calibration_risk.loc[
                    cal_ids, f"risk_h{horizon}"
                ].to_numpy()
                raw_test = test_risk.loc[
                    test_ids, f"risk_h{horizon}"
                ].to_numpy()
                calibrator, calibration_method = exp.fit_platt_calibrator(
                    y_cal,
                    raw_cal,
                )
                p_cal = exp.apply_probability_calibrator(
                    calibrator,
                    raw_cal,
                )
                p_test = exp.apply_probability_calibrator(
                    calibrator,
                    raw_test,
                )
                threshold = exp.select_threshold(y_cal, p_cal)
                metrics = exp.score_binary(y_test, p_test, threshold)
                metrics.update({
                    "test_session": test_session,
                    "model": model_name,
                    "horizon_s": horizon,
                    "calibration_source": calibration_source,
                    "calibration_method": calibration_method,
                    "fit_anchor_rows": int(len(fit)),
                    "fit_person_period_rows": int(len(fit_pp)),
                    "calibration_anchor_rows": int(len(calibration)),
                    "test_anchor_rows": int(len(test_ids)),
                    "n_used_features": int(len(num) + len(cat)),
                    "removed_all_missing_features": "|".join(removed),
                })
                result_rows.append(metrics)

                prediction = test.loc[
                    test_ids,
                    [
                        "session", "timestamp", "temporal_segment_id",
                        "mission_leg_id", "time_to_onset_s", label,
                    ],
                ].copy()
                prediction["model"] = model_name
                prediction["horizon_s"] = horizon
                prediction["probability"] = p_test
                prediction["threshold"] = threshold
                prediction["prediction"] = (p_test >= threshold).astype(int)
                prediction_rows.append(prediction)

            coefficients = coefficient_table(pipe, model_name)
            if not coefficients.empty:
                coefficients["test_session"] = test_session
                coefficient_frames.append(coefficients)

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError(
            "No hazard folds were produced. Check event counts and calibration eligibility."
        )
    predictions = (
        pd.concat(prediction_rows, ignore_index=True)
        if prediction_rows else pd.DataFrame()
    )
    coefficients = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames else pd.DataFrame()
    )

    results.to_csv(out_dir / "01_hazard_loso_results.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(out_dir / "02_hazard_predictions.csv", index=False)
    if not coefficients.empty:
        coefficients.to_csv(out_dir / "03_hazard_coefficients.csv", index=False)

    aggregate = results.groupby(
        ["horizon_s", "model"],
        as_index=False,
    ).agg(
        folds=("test_session", "nunique"),
        pr_auc_evaluable_folds=("pr_auc", "count"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", "std"),
        f1_positive_mean=("f1_positive", "mean"),
        recall_mean=("recall", "mean"),
        brier_mean=("brier", "mean"),
    )
    aggregate.to_csv(out_dir / "04_hazard_aggregate.csv", index=False)

    support_by_horizon = {}
    for horizon in (5, 10, 20):
        horizon_results = results[results["horizon_s"].eq(horizon)]
        score_table = horizon_results.pivot(
            index="test_session",
            columns="model",
            values="pr_auc",
        )
        proposed = "ProvenanceConditionedHazard"
        ablations = ["TelemetryHazard", "UngatedGraphHazard"]
        horizon_support = {
            "provenance_pr_auc_mean": None,
            "supported_over_both_ablations": False,
            "paired_positive_folds": {},
            "paired_pr_auc_wins": {},
        }
        proposed_scores = pd.to_numeric(
            score_table.get(proposed, pd.Series(dtype=float)),
            errors="coerce",
        )
        if proposed_scores.notna().any():
            horizon_support["provenance_pr_auc_mean"] = float(
                proposed_scores.mean()
            )

        mean_wins = []
        majority_wins = []
        for ablation in ablations:
            ablation_scores = pd.to_numeric(
                score_table.get(ablation, pd.Series(dtype=float)),
                errors="coerce",
            )
            paired = pd.concat(
                [proposed_scores.rename("proposed"),
                 ablation_scores.rename("ablation")],
                axis=1,
            ).dropna()
            wins = int((paired["proposed"] > paired["ablation"]).sum())
            paired_folds = int(len(paired))
            horizon_support["paired_positive_folds"][ablation] = paired_folds
            horizon_support["paired_pr_auc_wins"][ablation] = wins
            mean_wins.append(
                paired_folds > 0
                and float(paired["proposed"].mean())
                > float(paired["ablation"].mean())
            )
            majority_wins.append(
                paired_folds > 0 and wins > paired_folds / 2
            )
        horizon_support["supported_over_both_ablations"] = bool(
            all(mean_wins) and all(majority_wins)
        )
        support_by_horizon[str(horizon)] = horizon_support

    manifest = {
        "tested_hypothesis": "provenance-conditioned discrete-time degradation-onset hazard",
        "outer_protocol": "leave-one-session-out",
        "hazard_bins_s": list(range(1, MAX_HORIZON + 1)),
        "reported_horizons_s": [5, 10, 20],
        "primary_comparison": [
            "TelemetryHazard",
            "UngatedGraphHazard",
            "ProvenanceConditionedHazard",
        ],
        "interpretation": (
            "The proposed model is supported only if provenance conditioning "
            "improves session-level onset metrics over both hazard ablations."
        ),
        "support_by_horizon": support_by_horizon,
        "overall_claim_supported": bool(
            all(
                item["supported_over_both_ablations"]
                for item in support_by_horizon.values()
            )
        ),
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Provenance-conditioned hazard experiment completed.")
    print(aggregate.to_string(index=False))
    print(
        "Overall provenance superiority claim supported:",
        manifest["overall_claim_supported"],
    )
    print(f"Results: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
