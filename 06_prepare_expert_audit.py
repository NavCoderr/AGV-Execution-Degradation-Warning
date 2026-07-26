from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
NEGATIVE_CONTROLS_PER_SESSION = 6
WINDOW_BEFORE_S = 15
WINDOW_AFTER_S = 5


def load_experiment_module(path: Path):
    spec = importlib.util.spec_from_file_location("agv_experiments", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment utilities: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_best_h5_configuration(
    summary: pd.DataFrame,
) -> tuple[str, str]:
    learned = summary[
        summary["horizon_s"].eq(5)
        & summary["variant"].eq("all_sessions")
        & ~summary["model"].isin(["AlwaysNegative", "PersistenceH0"])
    ].copy()
    learned["pr_auc_mean"] = pd.to_numeric(
        learned["pr_auc_mean"],
        errors="coerce",
    )
    learned["macro_f1_mean"] = pd.to_numeric(
        learned["macro_f1_mean"],
        errors="coerce",
    )
    learned = learned.sort_values(
        ["pr_auc_mean", "macro_f1_mean"],
        ascending=[False, False],
    )
    if learned.empty:
        raise RuntimeError("No learned 5-s configuration found in summary.")
    best = learned.iloc[0]
    return str(best["model"]), str(best["feature_set"])


def alert_tradeoffs(
    predictions: pd.DataFrame,
    model: str,
    feature_set: str,
    exp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = predictions[
        predictions["horizon_s"].eq(5)
        & predictions["variant"].eq("all_sessions")
        & predictions["model"].eq(model)
        & predictions["feature_set"].eq(feature_set)
    ].copy()
    if selected.empty:
        raise RuntimeError(
            f"No predictions found for {model} / {feature_set} at 5 s."
        )
    selected["timestamp"] = pd.to_datetime(selected["timestamp"])

    profiles = {
        "sensitive_0.8x": 0.8,
        "learned_1.0x": 1.0,
        "conservative_1.2x": 1.2,
    }
    fold_rows: list[dict] = []
    for session, fold in selected.groupby("session", sort=True):
        for profile, multiplier in profiles.items():
            work = fold.copy()
            learned_threshold = float(work["threshold"].iloc[0])
            threshold = float(np.clip(learned_threshold * multiplier, 0.02, 0.95))
            work["alert"] = exp.segmented_alert_policy(work, on=threshold)
            metrics = exp.alert_metrics(work, "label_h5", 5)
            metrics.update(
                {
                    "test_session": session,
                    "model": model,
                    "feature_set": feature_set,
                    "operating_point": profile,
                    "learned_threshold": learned_threshold,
                    "applied_threshold": threshold,
                    "evaluated_hours": float(len(work) / 3600.0),
                }
            )
            fold_rows.append(metrics)

    per_fold = pd.DataFrame(fold_rows)
    aggregate_rows: list[dict] = []
    for profile, group in per_fold.groupby("operating_point", sort=True):
        events = int(group["events"].sum())
        warned = int(group["warned_events"].sum())
        false_alerts = int(group["false_alert_episodes"].sum())
        hours = float(group["evaluated_hours"].sum())
        aggregate_rows.append(
            {
                "horizon_s": 5,
                "model": model,
                "feature_set": feature_set,
                "operating_point": profile,
                "folds": int(group["test_session"].nunique()),
                "events": events,
                "warned_events": warned,
                "event_recall": float(warned / events) if events else np.nan,
                "alert_episodes": int(group["alert_episodes"].sum()),
                "false_alert_episodes": false_alerts,
                "false_alerts_per_hour": (
                    float(false_alerts / hours) if hours > 0 else np.nan
                ),
                "median_fold_lead_s": float(
                    pd.to_numeric(group["median_lead_s"], errors="coerce").median()
                ),
            }
        )
    return per_fold, pd.DataFrame(aggregate_rows)


def separated_sample(
    candidates: pd.DataFrame,
    count: int,
    random_state: int,
    minimum_separation_s: int = 30,
) -> pd.DataFrame:
    if candidates.empty or count <= 0:
        return candidates.head(0)
    shuffled = candidates.sample(
        frac=1.0,
        random_state=random_state,
    )
    selected_indices: list[int] = []
    selected_times: list[pd.Timestamp] = []
    for index, row in shuffled.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        if all(
            abs((timestamp - previous).total_seconds()) >= minimum_separation_s
            for previous in selected_times
        ):
            selected_indices.append(index)
            selected_times.append(timestamp)
        if len(selected_indices) >= count:
            break
    return candidates.loc[selected_indices].sort_values("timestamp")


def build_audit_cases(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = data[data["event_onset"].eq(1)].copy()
    events["automatic_case_type"] = "detected_onset"
    events["case_time"] = events["timestamp"]

    negative_parts: list[pd.DataFrame] = []
    negative_pool = data[
        data["at_risk"].eq(1)
        & data["label_h5"].eq(0)
        & data["telemetry_fresh"].eq(1)
        & data["active_mission"].eq(1)
        & data["external_hold"].eq(0)
    ].copy()
    for position, (session, group) in enumerate(
        negative_pool.groupby("session", sort=True)
    ):
        sampled = separated_sample(
            group,
            NEGATIVE_CONTROLS_PER_SESSION,
            RANDOM_STATE + position,
        ).copy()
        sampled["automatic_case_type"] = "negative_control"
        sampled["case_time"] = sampled["timestamp"]
        negative_parts.append(sampled)

    controls = (
        pd.concat(negative_parts, ignore_index=False)
        if negative_parts
        else negative_pool.head(0)
    )
    cases = pd.concat([events, controls], ignore_index=True)
    cases = cases.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    cases["case_id"] = [
        f"CASE_{position:03d}" for position in range(1, len(cases) + 1)
    ]

    key_columns = [
        "case_id",
        "automatic_case_type",
        "session",
        "case_time",
        "mission_leg_id",
        "temporal_segment_id",
    ]
    key = cases[key_columns].copy()

    # The reviewer form intentionally hides the automatic case type.
    form = cases[
        [
            "case_id",
            "session",
            "case_time",
            "mission_leg_id",
        ]
    ].copy()
    form["expert_label"] = ""
    form["expert_confidence"] = ""
    form["expert_reason"] = ""
    form["reviewer_id"] = ""
    form["review_date"] = ""
    return key, form


def build_audit_windows(
    data: pd.DataFrame,
    key: pd.DataFrame,
) -> pd.DataFrame:
    selected_columns = [
        "session",
        "timestamp",
        "mission_leg_id",
        "temporal_segment_id",
        "active_mission",
        "external_hold",
        "telemetry_available",
        "observed_sample",
        "short_causal_hold",
        "seconds_since_observed",
        "speed_mps",
        "speed_mean_10s",
        "stop_share_10s",
        "power_w",
        "current_ma",
        "voltage_mv",
        "soc",
        "wheel_mean",
        "requested_speed_mps",
        "command_mismatch_share_10s",
        "target_node",
        "target_reached",
        "graph_remaining_m",
        "graph_progress_5s_m",
        "euclidean_remaining_m",
        "euclidean_progress_5s_m",
        "route_source",
        "mission_state",
    ]
    selected_columns = [
        column for column in selected_columns if column in data.columns
    ]

    windows: list[pd.DataFrame] = []
    for row in key.itertuples(index=False):
        case_time = pd.Timestamp(row.case_time)
        group = data[
            data["session"].eq(row.session)
            & data["timestamp"].between(
                case_time - pd.Timedelta(seconds=WINDOW_BEFORE_S),
                case_time + pd.Timedelta(seconds=WINDOW_AFTER_S),
            )
        ][selected_columns].copy()
        if group.empty:
            continue
        group.insert(0, "case_id", row.case_id)
        group.insert(
            2,
            "relative_time_s",
            (group["timestamp"] - case_time).dt.total_seconds(),
        )
        windows.append(group)
    return (
        pd.concat(windows, ignore_index=True)
        if windows
        else pd.DataFrame()
    )


def summarize_completed_audit(
    form: pd.DataFrame,
    key: pd.DataFrame,
) -> pd.DataFrame:
    reviewed = form[
        form["expert_label"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if reviewed.empty:
        return pd.DataFrame()
    merged = reviewed.merge(key, on="case_id", how="left", validate="one_to_one")
    summary = (
        merged.groupby(
            ["automatic_case_type", "expert_label"],
            dropna=False,
        )
        .size()
        .rename("cases")
        .reset_index()
    )
    summary["reviewed_cases_total"] = int(len(reviewed))
    summary["unreviewed_cases"] = int(len(form) - len(reviewed))
    return summary


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "outputs" / "harmonized_graph_mission_state.csv"
    base_script = script_dir / "02_run_scientific_experiments.py"
    summary_file = (
        script_dir
        / "outputs"
        / "pre_onset_experiments"
        / "08_pre_onset_summary.csv"
    )
    predictions_file = (
        script_dir
        / "outputs"
        / "pre_onset_experiments"
        / "09_pre_onset_predictions.csv"
    )
    out_dir = script_dir / "outputs" / "acceptance_audits"

    required_files = [
        data_file,
        base_script,
        summary_file,
        predictions_file,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Required files are missing:\n" + "\n".join(missing_files)
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = load_experiment_module(base_script)
    data = pd.read_csv(data_file, parse_dates=["timestamp"], low_memory=False)
    summary = pd.read_csv(summary_file)
    predictions = pd.read_csv(
        predictions_file,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    model, feature_set = choose_best_h5_configuration(summary)
    alert_per_fold, alert_aggregate = alert_tradeoffs(
        predictions,
        model,
        feature_set,
        exp,
    )
    alert_per_fold.to_csv(
        out_dir / "01_alert_tradeoff_per_fold.csv",
        index=False,
    )
    alert_aggregate.to_csv(
        out_dir / "02_alert_tradeoff_aggregate.csv",
        index=False,
    )

    key, blank_form = build_audit_cases(data)
    windows = build_audit_windows(data, key)
    key.to_csv(out_dir / "03_expert_audit_key.csv", index=False)
    windows.to_csv(out_dir / "05_expert_audit_windows.csv", index=False)

    form_file = out_dir / "04_expert_audit_form.csv"
    if form_file.exists():
        existing_form = pd.read_csv(form_file, keep_default_na=False)
        expected_ids = set(blank_form["case_id"])
        existing_ids = set(existing_form.get("case_id", []))
        if existing_ids != expected_ids:
            raise RuntimeError(
                "Existing expert form does not match the current dataset. "
                "Move it to a safe location before regenerating the audit."
            )
        form = existing_form
    else:
        blank_form.to_csv(form_file, index=False)
        form = blank_form

    completed_summary = summarize_completed_audit(form, key)
    if not completed_summary.empty:
        completed_summary.to_csv(
            out_dir / "06_completed_expert_audit_summary.csv",
            index=False,
        )

    manifest = {
        "best_h5_model": model,
        "best_h5_feature_set": feature_set,
        "alert_operating_points": [
            "0.8 x learned threshold",
            "1.0 x learned threshold",
            "1.2 x learned threshold",
        ],
        "detected_onset_cases": int(
            key["automatic_case_type"].eq("detected_onset").sum()
        ),
        "negative_control_cases": int(
            key["automatic_case_type"].eq("negative_control").sum()
        ),
        "expert_form_is_blinded_to_automatic_case_type": True,
        "window_before_s": WINDOW_BEFORE_S,
        "window_after_s": WINDOW_AFTER_S,
        "allowed_expert_labels": {
            "detected onset interpretation": [
                "confirmed_degradation",
                "explained_stop",
                "not_degradation",
                "uncertain",
            ],
            "negative control interpretation": [
                "valid_negative",
                "missed_degradation",
                "uncertain",
            ],
        },
        "paper_claim_rule": (
            "Do not claim expert validation until the form is independently "
            "completed and the completed audit summary is generated."
        ),
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Acceptance audits prepared.")
    print("\nAlert trade-off:")
    print(alert_aggregate.to_string(index=False))
    print(
        f"\nExpert audit: {len(key)} blinded cases; "
        f"form: {form_file.resolve()}"
    )
    if completed_summary.empty:
        print(
            "The expert form is not completed yet. No expert-validation "
            "claim can be made."
        )
    else:
        print("\nCompleted expert-audit summary:")
        print(completed_summary.to_string(index=False))


if __name__ == "__main__":
    main()
