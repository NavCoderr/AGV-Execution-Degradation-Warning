from __future__ import annotations


import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# Direct-run configuration (no argparse)
DATA_FILE = Path("outputs/harmonized_graph_mission_state.csv")
OUT_DIR = Path("outputs/scientific_experiments")
FAST_MODE = False

BASE = [
    "speed_mps", "speed_mean_5s", "speed_mean_10s", "speed_mean_30s",
    "speed_std_10s", "stop_share_5s", "stop_share_10s", "stop_share_30s",
    "power_w", "power_mean_10s", "power_std_10s", "current_ma", "voltage_mv",
    "soc", "soc_drop_10s", "soc_drop_30s", "wheel_mean", "wheel_abs_diff",
    "command_mismatch_share_10s", "command_mismatch_share_30s",
    "position_confidence", "seconds_since_observed", "observed_sample",
    "telemetry_available", "short_causal_hold",
]
GRAPH = [
    "graph_remaining_m", "route_completion", "graph_progress_1s_m",
    "graph_progress_3s_m", "graph_progress_5s_m", "graph_progress_10s_m",
    "graph_progress_30s_m", "graph_progress_rate_mps", "edge_progress_fraction",
    "edge_remaining_m", "edge_lateral_error_m", "edge_dwell_s", "node_degree",
    "node_corridor", "node_intersection", "node_station",
]
EUCLID = [
    "euclidean_remaining_m", "euclidean_progress_5s_m",
    "euclidean_progress_rate_mps",
]
CAT = ["route_source"]

SOC_FEATURES = {"soc", "soc_drop_10s", "soc_drop_30s"}


def preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat),
    ], sparse_threshold=0.0)


def model_dict(fast: bool) -> dict[str, object]:
    if fast:
        return {
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=50,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        }
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    thresholds = np.arange(0.10, 0.91, 0.02)
    scores = [
        f1_score(
    y,
    (p >= t).astype(int),
    labels=[0, 1],
    average="macro",
    zero_division=0,
)
        for t in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def score_binary(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(
    f1_score(
        y,
        pred,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_positive": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def split_fit_calibration(train: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Choose a whole calibration session using training data only.

    The selected session must contain enough positive and negative examples.
    Among valid sessions, choose the positive rate closest to the complete
    training set, avoiding an oversized calibration split.
    """
    overall_rate = float(train[label].mean())
    candidates = []
    for session, group in train.groupby("session"):
        positives = int(group[label].sum())
        negatives = int(len(group) - positives)
        if len(group) >= 100 and positives >= 20 and negatives >= 20:
            candidates.append((abs(float(group[label].mean()) - overall_rate), len(group), session))
    if candidates:
        _, _, cal_session = sorted(candidates)[0]
        fit = train[train["session"] != cal_session].copy()
        cal = train[train["session"] == cal_session].copy()
        if fit[label].nunique() == 2:
            return fit, cal, f"session:{cal_session}"

    # Fall back to a chronological blocked split inside every development
    # session. This preserves temporal order and avoids randomly mixing
    # adjacent seconds between fitting and calibration.
    for fraction in (0.20, 0.30, 0.40):
        fit_parts = []
        cal_parts = []
        for _, group in train.groupby("session", sort=False):
            group = group.sort_values("timestamp")
            cut = max(1, int(np.floor(len(group) * (1.0 - fraction))))
            if cut >= len(group):
                continue
            fit_parts.append(group.iloc[:cut])
            cal_parts.append(group.iloc[cut:])
        if not fit_parts or not cal_parts:
            continue
        fit = pd.concat(fit_parts).copy()
        cal = pd.concat(cal_parts).copy()
        if fit[label].nunique() == 2 and cal[label].nunique() == 2:
            return fit, cal, f"blocked_tail_{int(fraction * 100)}pct"

    raise RuntimeError(
        "No leakage-safe calibration split contains both classes. "
        "Collect more event-containing sessions or use inner session-wise OOF calibration."
    )


def fit_platt_calibrator(y: np.ndarray, p_raw: np.ndarray):
    """Fit sigmoid/Platt probability calibration on held-out development data."""
    y = np.asarray(y, dtype=int)
    p_raw = np.asarray(p_raw, dtype=float)
    if len(np.unique(y)) < 2 or int(y.sum()) < 5 or int((1 - y).sum()) < 5:
        return None, "identity_insufficient_calibration_events"
    log_odds = np.log(
        np.clip(p_raw, 1e-6, 1 - 1e-6)
        / np.clip(1 - p_raw, 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    calibrator.fit(log_odds, y)
    return calibrator, "platt_sigmoid"


def apply_probability_calibrator(calibrator, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.asarray(p_raw, dtype=float)
    if calibrator is None:
        return np.clip(p_raw, 0.0, 1.0)
    log_odds = np.log(
        np.clip(p_raw, 1e-6, 1 - 1e-6)
        / np.clip(1 - p_raw, 1e-6, 1 - 1e-6)
    ).reshape(-1, 1)
    return calibrator.predict_proba(log_odds)[:, 1]


def usable_features_from_fit(
    fit: pd.DataFrame,
    requested_num: list[str],
    requested_cat: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Remove columns that are entirely missing in the fit partition only."""
    num = []
    cat = []
    removed = []
    for column in dict.fromkeys(requested_num):
        if column in fit.columns and pd.to_numeric(fit[column], errors="coerce").notna().any():
            num.append(column)
        else:
            removed.append(column)
    for column in dict.fromkeys(requested_cat):
        if column in fit.columns and fit[column].notna().any():
            cat.append(column)
        else:
            removed.append(column)
    return num, cat, removed


def alert_policy(prob: np.ndarray, on: float, off: float | None = None,
                 k: int = 2, n: int = 3, cooldown: int = 5) -> np.ndarray:
    # Hysteresis requires the off threshold to be below the on threshold.
    if off is None:
        off = max(0.02, min(on * 0.50, on - 0.02))
    state = False
    cool = 0
    history: list[float] = []
    alerts: list[int] = []
    for value in prob:
        history.append(float(value))
        history = history[-n:]
        if cool > 0:
            cool -= 1
        if not state and cool == 0 and sum(v >= on for v in history) >= k:
            state = True
        elif state and value <= off:
            state = False
            cool = cooldown
        alerts.append(int(state))
    return np.asarray(alerts, dtype=int)


def segmented_alert_policy(frame: pd.DataFrame, on: float) -> np.ndarray:
    """Apply the stateful alert policy separately to every contiguous leg."""
    work = frame.copy()
    work["_row_order"] = np.arange(len(work))
    sort_columns = [
        column for column in
        ["session", "temporal_segment_id", "mission_leg_id", "timestamp"]
        if column in work.columns
    ]
    work = work.sort_values(sort_columns)
    result = pd.Series(0, index=work.index, dtype=int)
    group_columns = [
        column for column in
        ["session", "temporal_segment_id", "mission_leg_id"]
        if column in work.columns
    ]
    grouped = work.groupby(group_columns, sort=False, dropna=False) if group_columns else [(None, work)]
    for _, group in grouped:
        group = group.sort_values("timestamp")
        breaks = group["timestamp"].diff().dt.total_seconds().ne(1)
        contiguous_id = breaks.cumsum()
        for _, segment in group.groupby(contiguous_id, sort=False):
            result.loc[segment.index] = alert_policy(
                segment["probability"].to_numpy(dtype=float),
                on=on,
            )
    return result.reindex(frame.index).to_numpy(dtype=int)


def alert_metrics(frame: pd.DataFrame, label: str, horizon: int) -> dict:
    frame = frame.sort_values("timestamp").copy()
    starts = frame["alert"].eq(1) & frame["alert"].shift(fill_value=0).eq(0)
    # Eligible rows are the actual monitored exposure; timestamp span would
    # incorrectly include inactive gaps removed by the at-risk filter.
    duration_h = max(len(frame) / 3600.0, 1e-9)
    if horizon > 0 and "time_to_onset_s" in frame:
        event_times = (
            frame.loc[frame["time_to_onset_s"].between(1, horizon), "timestamp"]
            + pd.to_timedelta(
                frame.loc[frame["time_to_onset_s"].between(1, horizon), "time_to_onset_s"],
                unit="s",
            )
        )
        onsets = sorted(pd.Series(event_times).dropna().drop_duplicates().tolist())
    else:
        event_onsets = frame[label].eq(1) & frame[label].shift(fill_value=0).eq(0)
        onsets = frame.loc[event_onsets, "timestamp"].tolist()

    start_times = frame.loc[starts, "timestamp"].tolist()
    matched_start = {
        start: any(start <= onset <= start + pd.Timedelta(seconds=max(horizon, 1))
                   for onset in onsets)
        for start in start_times
    }
    false_starts = sum(not matched for matched in matched_start.values())
    leads: list[float] = []
    warned = 0
    for onset in onsets:
        before = frame[
            (frame["timestamp"] <= onset)
            & (frame["timestamp"] >= onset - pd.Timedelta(seconds=max(horizon, 1)))
            & frame["alert"].eq(1)
        ]
        if not before.empty:
            warned += 1
            leads.append(float((onset - before["timestamp"].min()).total_seconds()))
    return {
        "alert_episodes": int(starts.sum()),
        "false_alert_episodes": int(false_starts),
        "false_alerts_per_hour": float(false_starts / duration_h),
        "events": int(len(onsets)),
        "warned_events": int(warned),
        "event_recall": float(warned / len(onsets)) if onsets else np.nan,
        "median_lead_s": float(np.median(leads)) if leads else np.nan,
        "pct_events_lead_ge_2s": float(np.mean(np.asarray(leads) >= 2)) if leads else np.nan,
        "pct_events_lead_ge_5s": float(np.mean(np.asarray(leads) >= 5)) if leads else np.nan,
    }


def exact_1hz_check(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session, group in data.groupby("session"):
        diff = group.sort_values("timestamp")["timestamp"].diff().dt.total_seconds().dropna()
        rows.append({
            "session": session,
            "rows": int(len(group)),
            "all_intervals_1s": bool((diff == 1).all()),
            "min_interval_s": float(diff.min()) if len(diff) else np.nan,
            "max_interval_s": float(diff.max()) if len(diff) else np.nan,
            "observed_rows": int(group["observed_sample"].eq(1).sum()),
            "short_hold_rows": int(group["short_causal_hold"].eq(1).sum()),
            "unavailable_rows": int(group["telemetry_available"].eq(0).sum()),
        })
    return pd.DataFrame(rows)


def inject_gap(frame: pd.DataFrame, num: list[str], cat: list[str], gap_s: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Inject causal telemetry gaps consistent with the 1-Hz preprocessing rule."""
    corrupted = frame[num + cat].copy()
    rng = np.random.default_rng(RANDOM_STATE + gap_s)
    injected = np.zeros(len(corrupted), dtype=bool)
    unavailable = np.zeros(len(corrupted), dtype=bool)
    if len(corrupted) <= gap_s + 1:
        return corrupted, injected, unavailable
    count = max(1, len(corrupted) // 250)
    possible_starts = np.arange(1, len(corrupted) - gap_s)
    if len(possible_starts) == 0:
        return corrupted, injected, unavailable
    starts = rng.choice(
        possible_starts,
        size=min(count, len(possible_starts)),
        replace=False,
    )
    for start in starts:
        stop = start + gap_s
        injected[start:stop] = True
        previous = corrupted.iloc[start - 1].copy()
        for offset, row in enumerate(range(start, stop), start=1):
            row_index = corrupted.index[row]
            if "observed_sample" in corrupted:
                corrupted.loc[row_index, "observed_sample"] = 0
            if "seconds_since_observed" in corrupted:
                corrupted.loc[row_index, "seconds_since_observed"] = offset
            if offset <= 2:
                # Causally hold the last pre-gap representation. Do not retain
                # the original values from inside the artificially hidden gap.
                corrupted.loc[row_index, num + cat] = previous[num + cat]
                if "observed_sample" in corrupted:
                    corrupted.loc[row_index, "observed_sample"] = 0
                if "seconds_since_observed" in corrupted:
                    corrupted.loc[row_index, "seconds_since_observed"] = offset
                if "short_causal_hold" in corrupted:
                    corrupted.loc[row_index, "short_causal_hold"] = 1
                if "telemetry_available" in corrupted:
                    corrupted.loc[row_index, "telemetry_available"] = 1
            else:
                unavailable[row] = True
                corrupted.loc[row_index, num] = np.nan
                corrupted.loc[row_index, cat] = np.nan
                if "short_causal_hold" in corrupted:
                    corrupted.loc[row_index, "short_causal_hold"] = 0
                if "telemetry_available" in corrupted:
                    corrupted.loc[row_index, "telemetry_available"] = 0
                if "observed_sample" in corrupted:
                    corrupted.loc[row_index, "observed_sample"] = 0
                if "seconds_since_observed" in corrupted:
                    corrupted.loc[row_index, "seconds_since_observed"] = offset
    return corrupted, injected, unavailable


def engineering_baselines(test: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    """Simple, transparent baselines using only information available now."""
    baselines: dict[str, np.ndarray] = {
        "AlwaysNegative": np.zeros(len(test), dtype=float),
    }
    if horizon > 0 and "current_degraded" in test.columns:
        baselines["PersistenceH0"] = (
            pd.to_numeric(test["current_degraded"], errors="coerce")
            .fillna(0)
            .astype(int)
            .to_numpy(dtype=float)
        )
    if horizon == 5:
        if {"stop_share_10s", "graph_progress_10s_m"}.issubset(test.columns):
            baselines["RuleGraph"] = (
                test["stop_share_10s"].ge(0.60)
                & test["graph_progress_10s_m"].fillna(-1).lt(0.03)
            ).astype(float).to_numpy()
        if {"stop_share_10s", "euclidean_progress_10s_m"}.issubset(test.columns):
            baselines["RuleEuclidean"] = (
                test["stop_share_10s"].ge(0.60)
                & test["euclidean_progress_10s_m"].fillna(-1).lt(0.03)
            ).astype(float).to_numpy()
    return baselines


def make_final_figures(results: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Primary 5-s comparison. PR-AUC is used because the positive onset class
    # is rare; negative-only held-out sessions are not evaluable for PR-AUC
    # and remain part of the false-alarm and Brier-score analysis.
    selection = results[
        (results["variant"] == "all_sessions")
        & (results["horizon_s"] == 5)
        & results["feature_set"].isin(
            ["full_graph", "full_graph_no_soc", "euclidean_only", "no_graph"]
        )
        & results["model"].isin(
            ["LogisticRegression", "ExtraTrees", "RandomForest", "HistGradientBoosting"]
        )
    ]
    if not selection.empty:
        model_order = [
            "LogisticRegression", "ExtraTrees",
            "RandomForest", "HistGradientBoosting",
        ]
        feature_order = [
            "no_graph", "euclidean_only",
            "full_graph", "full_graph_no_soc",
        ]
        feature_labels = {
            "no_graph": "Base context",
            "euclidean_only": "Base + Euclidean",
            "full_graph": "Base + Euc. + graph",
            "full_graph_no_soc": "Base + Euc. + graph\n(no SOC)",
        }
        table = (
            selection.groupby(["model", "feature_set"])["pr_auc"]
            .mean()
            .unstack("feature_set")
            .reindex(index=model_order, columns=feature_order)
        )
        error = (
            selection.groupby(["model", "feature_set"])["pr_auc"]
            .std()
            .unstack("feature_set")
            .reindex(index=model_order, columns=feature_order)
        )
        table = table.rename(columns=feature_labels)
        error = error.rename(columns=feature_labels)
        model_labels = {
            "LogisticRegression": "Logistic\nRegression",
            "ExtraTrees": "ExtraTrees",
            "RandomForest": "Random\nForest",
            "HistGradientBoosting": "Histogram\nBoosting",
        }

        ax = table.plot(
            kind="bar",
            yerr=error,
            capsize=3,
            figsize=(6.4, 4.4),
            width=0.82,
        )

        ax.set_ylabel("Mean PR-AUC")
        ax.set_xlabel("Model")
        ax.set_ylim(0, 1)

        ax.set_xticklabels(
            [model_labels[model] for model in model_order],
            rotation=0,
            ha="center",
        )

        handles, labels = ax.get_legend_handles_labels()

        legend_order = [0, 2, 1, 3]

        legend = ax.legend(
            [handles[index] for index in legend_order],
            [labels[index] for index in legend_order],
            title="Feature representation",
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=8,
            title_fontsize=8,
            frameon=True,
        )

        figure = ax.get_figure()

        figure.subplots_adjust(
            top=0.76,
            bottom=0.18,
        )

        figure.savefig(
            fig_dir / "final_h5_model_ablation.png",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=(legend,),
        )

        plt.close(figure)

    # Plot one mission leg from the median-performing positive fold.
    ranked = (
        selection.groupby(["model", "feature_set"])["pr_auc"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )
    q = pd.DataFrame()
    best_model = best_feature = None
    if not ranked.empty:
        best_model, best_feature = ranked.index[0]
        q = predictions[
            (predictions["variant"] == "all_sessions")
            & (predictions["feature_set"] == best_feature)
            & (predictions["model"] == best_model)
            & (predictions["horizon_s"] == 5)
        ].copy()
    if not q.empty:
        q["timestamp"] = pd.to_datetime(q["timestamp"])
        fold_scores = []
        for session_name, fold in q.groupby("session"):
            y_fold = fold["label_h5"].dropna().astype(int)
            if y_fold.nunique() < 2:
                continue
            p_fold = fold.loc[y_fold.index, "probability"].astype(float)
            fold_scores.append((
                session_name,
                average_precision_score(y_fold, p_fold),
            ))
        fold_scores.sort(key=lambda item: item[1])
        session = fold_scores[len(fold_scores) // 2][0] if fold_scores else q["session"].iloc[0]
        g = q[q["session"] == session].sort_values("timestamp").copy()

        positive_legs = (
            g.groupby("mission_leg_id")["label_h5"].sum()
            .loc[lambda x: x > 0]
            .sort_values(ascending=False)
        )
        if not positive_legs.empty:
            g = g[g["mission_leg_id"].eq(positive_legs.index[0])].copy()

        all_onset_candidates = g.loc[
            g["time_to_onset_s"].notna(),
            ["timestamp", "time_to_onset_s"],
        ].copy()
        selected_onset_time = None
        if not all_onset_candidates.empty:
            candidate_times = (
                all_onset_candidates["timestamp"]
                + pd.to_timedelta(
                    all_onset_candidates["time_to_onset_s"],
                    unit="s",
                )
            ).drop_duplicates().sort_values()
            selected_onset_time = candidate_times.iloc[len(candidate_times) // 2]
            g = g[
                g["timestamp"].between(
                    selected_onset_time - pd.Timedelta(seconds=30),
                    selected_onset_time + pd.Timedelta(seconds=3),
                )
            ].copy()
        else:
            g = g.head(45).copy()

        threshold = float(g["threshold"].iloc[0])
        elapsed = (g["timestamp"] - g["timestamp"].min()).dt.total_seconds()
        contiguous = g["timestamp"].diff().dt.total_seconds().fillna(1).eq(1)
        probability = g["probability"].where(contiguous)

        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.plot(elapsed, probability, linewidth=1.8, label="Calibrated onset probability")
        if selected_onset_time is not None:
            row_onset_time = (
                g["timestamp"]
                + pd.to_timedelta(g["time_to_onset_s"], unit="s")
            )
            selected_target = (
    row_onset_time.eq(selected_onset_time)
    & g["label_h5"].fillna(0).eq(1)
)
        else:
            selected_target = g["label_h5"].fillna(0).eq(1)
        positive_elapsed = elapsed[selected_target]
        for position, second in enumerate(positive_elapsed):
            ax.axvspan(
                second,
                second + 1,
                alpha=0.18,
                color="tab:orange",
                label="Onset within next 5 s" if position == 0 else None,
            )
        ax.axhline(
            threshold,
            linestyle="--",
            color="tab:red",
            label=f"Fold threshold = {threshold:.2f}",
        )

        if selected_onset_time is not None:
            onset_elapsed = (
                selected_onset_time - g["timestamp"].min()
            ).total_seconds()
            if elapsed.min() <= onset_elapsed <= elapsed.max():
                ax.axvline(
                    onset_elapsed,
                    color="tab:orange",
                    linewidth=1.8,
                    label="Confirmed onset",
                )

        ax.set_ylabel("Calibrated probability")
        ax.set_xlabel("Elapsed time within mission leg (s)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Representative held-out 5-s onset forecast")
        ax.text(
            0.01, 0.98,
            f"{best_model}; {feature_labels.get(best_feature, best_feature)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(rect=(0, 0, 0.80, 1))
        plt.savefig(fig_dir / "final_calibrated_risk_case.png", dpi=250)
        plt.close()



def write_scientific_summaries(results: pd.DataFrame, data: pd.DataFrame, out_dir: Path) -> None:

    if results.empty:
        return
    group_cols = ["horizon_s", "variant", "feature_set", "model"]
    metrics = ["macro_f1", "pr_auc", "f1_positive", "recall", "brier"]
    agg = results.groupby(group_cols, dropna=False).agg(
        folds=("test_session", "nunique"),
        positive_test_folds=("positive_rate", lambda x: int((x > 0).sum())),
        pr_auc_evaluable_folds=("pr_auc", "count"),
        test_rows=("test_rows", "sum"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_sd": (m, "std") for m in metrics},
    ).reset_index()
    agg.to_csv(out_dir / "07_aggregate_results.csv", index=False)

    h5 = results[(results["horizon_s"] == 5) &
                 (results["variant"] == "all_sessions") &
                 results["model"].isin(["LogisticRegression", "ExtraTrees", "RandomForest", "HistGradientBoosting"])]
    rows = []
    for model, g in h5.groupby("model"):
        base = g[g["feature_set"] == "no_graph"][["test_session", "macro_f1", "pr_auc"]].rename(
            columns={"macro_f1":"base_macro_f1", "pr_auc":"base_pr_auc"})
        for feature_set in ["euclidean_only", "full_graph", "full_graph_no_soc"]:
            q = g[g["feature_set"] == feature_set][["test_session", "macro_f1", "pr_auc"]]
            merged = q.merge(base, on="test_session", how="inner")
            if merged.empty:
                continue
            dm = merged["macro_f1"] - merged["base_macro_f1"]
            dp = merged["pr_auc"] - merged["base_pr_auc"]
            valid_dp = dp.dropna()
            rows.append({
                "model": model, "comparison": f"{feature_set}-minus-no_graph",
                "folds": len(merged), "macro_f1_delta_mean": dm.mean(),
                "macro_f1_delta_median": dm.median(), "macro_f1_wins": int((dm>0).sum()),
                "macro_f1_ties": int((dm==0).sum()), "macro_f1_losses": int((dm<0).sum()),
                "pr_auc_evaluable_folds": int(valid_dp.size),
                "pr_auc_delta_mean": valid_dp.mean(), "pr_auc_delta_median": valid_dp.median(),
                "pr_auc_wins": int((valid_dp>0).sum()), "pr_auc_losses": int((valid_dp<0).sum()),
                "interpretation": "descriptive session-level comparison; n=5 folds, not a conclusive significance test",
            })
    pd.DataFrame(rows).to_csv(out_dir / "08_representation_deltas.csv", index=False)

    event_dist = data.groupby("session", as_index=False).agg(
        aligned_rows=("timestamp","size"),
        eligible_h5=("label_h5", lambda x: int(x.notna().sum())),
        positive_h5=("label_h5", lambda x: int((x==1).sum())),
        event_onsets=("event_onset","sum"),
    )
    event_dist.to_csv(out_dir / "09_event_distribution.csv", index=False)

    manifest = {
        "primary_unit_of_generalization": "recording session",
        "primary_protocol": "leave-one-session-out",
        "primary_task": "causal short-horizon degradation-onset forecasting",
        "strict_transition_task": "run 03_run_pre_onset_experiments.py",
        "new_classifier_claimed": False,
        "scientific_question": "Does mission/route context add value beyond telemetry and Euclidean proximity for genuine future onsets across unseen physical sessions?",
        "caution": "Five folds and one AGV/layout limit inferential and external validity.",
    }
    (out_dir / "10_experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    data_file = DATA_FILE
    out_dir = OUT_DIR
    fast_mode = FAST_MODE

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}. Run 01_build_scientific_dataset.py first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_file, parse_dates=["timestamp"], low_memory=False)
    required = {"session", "timestamp", "active_mission", "external_hold", "telemetry_fresh"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")

    hz = exact_1hz_check(data)
    hz.to_csv(out_dir / "01_exact_1hz_verification.csv", index=False)
    if not hz["all_intervals_1s"].all():
        bad = hz.loc[~hz["all_intervals_1s"], "session"].tolist()
        raise RuntimeError(f"Not exact 1-Hz for sessions: {bad}")

    variants = {
        "all_sessions": pd.Series(True, index=data.index),
        "annotated_sessions": data["source_annotation_quality"].ne("telemetry_only"),
        "recorded_route_only": data["route_source"].eq("recorded"),
    }
    full_graph_features = BASE + GRAPH + EUCLID
    feature_sets = {
        "full_graph": full_graph_features,
        "full_graph_no_soc": [c for c in full_graph_features if c not in SOC_FEATURES],
        "no_graph": BASE,
        "euclidean_only": BASE + EUCLID,
    }

    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    alert_rows: list[dict] = []
    gap_rows: list[dict] = []
    latency_rows: list[dict] = []

    for horizon in [0, 5, 10, 20]:
        label = f"label_h{horizon}"
        if label not in data:
            continue
        risk_mask = (
            data["at_risk"].eq(1)
            if horizon > 0 and "at_risk" in data.columns
            else pd.Series(True, index=data.index)
        )
        for variant_name, variant_mask in variants.items():
            if horizon != 5 and variant_name != "all_sessions":
                continue
            eligible = data[
                variant_mask
                & risk_mask
                & data["active_mission"].eq(1)
                & data["external_hold"].eq(0)
                & data["telemetry_fresh"].eq(1)
                & data[label].notna()
            ].copy()
            for feature_name, requested_features in feature_sets.items():
                if horizon != 5 and feature_name != "full_graph":
                    continue
                if variant_name != "all_sessions" and feature_name != "full_graph":
                    continue
                if feature_name == "full_graph_no_soc" and not (
                    horizon == 5 and variant_name == "all_sessions"
                ):
                    continue
                requested_num = [c for c in requested_features if c in eligible.columns]
                requested_cat = [c for c in CAT if c in eligible.columns]
                for test_session in sorted(eligible["session"].dropna().unique()):
                    test = eligible[eligible["session"] == test_session].copy()
                    train = eligible[eligible["session"] != test_session].copy()
                    if len(test) < 20 or train[label].nunique() < 2:
                        continue
                    try:
                        fit, calibration, calibration_source = split_fit_calibration(train, label)
                    except RuntimeError:
                        continue
                    if fit[label].nunique() < 2 or calibration[label].nunique() < 2:
                        continue
                    num, cat, removed_features = usable_features_from_fit(
                        fit, requested_num, requested_cat
                    )
                    if not num and not cat:
                        continue
                    for model_name, model in model_dict(fast_mode).items():
                        pipe = Pipeline([
                            ("prep", preprocessor(num, cat)),
                            ("model", model),
                        ])
                        start = time.perf_counter()
                        pipe.fit(fit[num + cat], fit[label].astype(int))
                        fit_seconds = time.perf_counter() - start

                        p_cal_raw = pipe.predict_proba(calibration[num + cat])[:, 1]
                        calibrator, calibration_method = fit_platt_calibrator(
                            calibration[label].astype(int).to_numpy(),
                            p_cal_raw,
                        )
                        p_cal = apply_probability_calibrator(calibrator, p_cal_raw)
                        threshold = select_threshold(calibration[label].astype(int).to_numpy(), p_cal)

                        start = time.perf_counter()
                        p_test_raw = pipe.predict_proba(test[num + cat])[:, 1]
                        p_test = apply_probability_calibrator(calibrator, p_test_raw)
                        infer_ms_row = (time.perf_counter() - start) * 1000 / len(test)
                        y_test = test[label].astype(int).to_numpy()

                        metrics = score_binary(y_test, p_test, threshold)
                        metrics.update({
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

                        prediction_columns = [
                            column for column in [
                                "session", "timestamp", "temporal_segment_id",
                                "mission_leg_id", "time_to_onset_s", label,
                            ]
                            if column in test.columns
                        ]
                        prediction = test[prediction_columns].copy()
                        prediction["horizon_s"] = horizon
                        prediction["variant"] = variant_name
                        prediction["feature_set"] = feature_name
                        prediction["model"] = model_name
                        prediction["probability"] = p_test
                        prediction["threshold"] = threshold
                        prediction["prediction"] = (p_test >= threshold).astype(int)
                        prediction["alert"] = segmented_alert_policy(
                            prediction,
                            on=threshold,
                        )
                        prediction_frames.append(prediction)

                        am = alert_metrics(prediction, label, horizon)
                        am.update({
                            "horizon_s": horizon,
                            "variant": variant_name,
                            "feature_set": feature_name,
                            "test_session": test_session,
                            "model": model_name,
                            "threshold": threshold,
                        })
                        alert_rows.append(am)

                        sample = test[num + cat].iloc[:min(500, len(test))]
                        repeats = 3 if fast_mode else 30
                        timings = []
                        for _ in range(repeats):
                            t0 = time.perf_counter_ns()
                            pipe.predict_proba(sample)
                            timings.append((time.perf_counter_ns() - t0) / 1e6 / len(sample))
                        latency_rows.append({
                            "horizon_s": horizon,
                            "variant": variant_name,
                            "feature_set": feature_name,
                            "test_session": test_session,
                            "model": model_name,
                            "p50_ms_row": float(np.percentile(timings, 50)),
                            "p95_ms_row": float(np.percentile(timings, 95)),
                            "p99_ms_row": float(np.percentile(timings, 99)),
                        })

                        if (horizon == 5 and variant_name == "all_sessions"
                                and feature_name == "full_graph" and model_name == "ExtraTrees"):
                            for gap_s in [1, 2, 3, 5, 10]:
                                corrupted, gap_mask, unavailable_mask = inject_gap(test, num, cat, gap_s)
                                p_gap_raw = pipe.predict_proba(corrupted)[:, 1]
                                p_gap = apply_probability_calibrator(calibrator, p_gap_raw)

                                overall_metric = score_binary(y_test, p_gap, threshold)
                                overall_metric.update({
                                    "test_session": test_session,
                                    "gap_s": gap_s,
                                    "injected_rows": int(gap_mask.sum()),
                                    "unavailable_rows": int(unavailable_mask.sum()),
                                    "prediction_coverage": float((~unavailable_mask).mean()),
                                    "model": model_name,
                                    "scope": "complete_test_session",
                                })
                                gap_rows.append(overall_metric)

                                if gap_mask.any():
                                    affected_metric = score_binary(
                                        y_test[gap_mask],
                                        p_gap[gap_mask],
                                        threshold,
                                    )
                                    affected_metric.update({
                                        "test_session": test_session,
                                        "gap_s": gap_s,
                                        "injected_rows": int(gap_mask.sum()),
                                        "unavailable_rows": int(unavailable_mask.sum()),
                                        "prediction_coverage": float((~unavailable_mask).mean()),
                                        "model": model_name,
                                        "scope": "injected_rows_only",
                                    })
                                    gap_rows.append(affected_metric)

        # Transparent non-learned baselines on the all-session protocol.
        eligible_baseline = data[
            risk_mask
            & data["active_mission"].eq(1)
            & data["external_hold"].eq(0)
            & data["telemetry_fresh"].eq(1)
            & data[label].notna()
        ].copy()
        for test_session in sorted(eligible_baseline["session"].dropna().unique()):
            test = eligible_baseline[eligible_baseline["session"] == test_session].copy()
            if len(test) < 20:
                continue
            y_test = test[label].astype(int).to_numpy()
            for baseline_name, p_test in engineering_baselines(test, horizon).items():
                threshold = 0.5
                metrics = score_binary(y_test, p_test, threshold)
                metrics.update({
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

                prediction_columns = [
                    column for column in [
                        "session", "timestamp", "temporal_segment_id",
                        "mission_leg_id", "time_to_onset_s", label,
                    ]
                    if column in test.columns
                ]
                prediction = test[prediction_columns].copy()
                prediction["horizon_s"] = horizon
                prediction["variant"] = "all_sessions"
                prediction["feature_set"] = "engineering_baseline"
                prediction["model"] = baseline_name
                prediction["probability"] = p_test
                prediction["threshold"] = threshold
                prediction["prediction"] = (p_test >= threshold).astype(int)
                prediction["alert"] = segmented_alert_policy(
                    prediction,
                    on=threshold,
                )
                prediction_frames.append(prediction)

                am = alert_metrics(prediction, label, horizon)
                am.update({
                    "horizon_s": horizon,
                    "variant": "all_sessions",
                    "feature_set": "engineering_baseline",
                    "test_session": test_session,
                    "model": baseline_name,
                    "threshold": threshold,
                })
                alert_rows.append(am)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(out_dir / "02_multi_horizon_loso_ablation.csv", index=False)
    predictions_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not predictions_df.empty:
        predictions_df.to_csv(out_dir / "03_predictions.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(out_dir / "04_alert_policy_metrics.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(out_dir / "05_gap_robustness.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(out_dir / "06_latency.csv", index=False)
    if not results_df.empty and not predictions_df.empty:
        make_final_figures(results_df, predictions_df, out_dir)
    write_scientific_summaries(results_df, data, out_dir)

    summary = {
        "processed_rows": int(len(data)),
        "sessions": int(data["session"].nunique()),
        "all_sessions_exact_1hz": bool(hz["all_intervals_1s"].all()),
        "model_result_rows": int(len(result_rows)),
        "alert_result_rows": int(len(alert_rows)),
        "gap_result_rows": int(len(gap_rows)),
        "status": "completed",
    }
    (out_dir / "00_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
