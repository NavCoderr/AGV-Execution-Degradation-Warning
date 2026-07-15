#!/usr/bin/env python3
"""

Inputs expected in --data_dir:
- 01_real_agv_live_LOW_SOC_STRESS*.csv
- 02_real_agv_live_HIGH_SOC_CONTROL*.csv
- Node_F3*.csv
- Edge_Distances3*.csv

"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def read_csv(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if d.shape[1] == 1:
        d = pd.read_csv(path, sep=";")
    return d


def first_existing(d: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    for name in names:
        if name in d.columns:
            return d[name]
    return pd.Series(default, index=d.index)


def bool01(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "y", "t"]
    ).astype(int)


def locate(data_dir: Path, tokens: List[str]) -> Path:
    matches = [
        p for p in data_dir.glob("*.csv")
        if all(t.lower() in p.name.lower() for t in tokens)
    ]
    if not matches:
        raise FileNotFoundError(f"No CSV found containing tokens: {tokens}")
    return sorted(matches, key=lambda p: len(p.name))[0]


def load_session(path: Path, role: str) -> Tuple[pd.DataFrame, int]:
    d = read_csv(path)
    raw_rows = len(d)

    if "agent" in d.columns:
        d = d[d["agent"].astype(str).str.upper().eq("REAL")].copy()

    d["timestamp"] = pd.to_datetime(
        first_existing(d, ["local_time", "timestamp", "time"]),
        errors="coerce"
    )
    d = (
        d.dropna(subset=["timestamp"])
         .sort_values("timestamp")
         .drop_duplicates("timestamp")
         .copy()
    )

    numeric = [
        "x", "y", "snap_node", "u", "v", "going_to_id", "target_reached",
        "speed_mps", "abs_speed", "battery_percent", "battery", "power_W",
        "current_mA", "cell_voltage_mV", "position_confidence", "series_index",
        "series_leg_start_node", "series_leg_goal_node", "series_leg_pred_time_s",
        "series_leg_distance_m", "requested_straight_speed_mps",
        "requested_corner_speed_mps"
    ]
    for c in numeric:
        d[c] = pd.to_numeric(first_existing(d, [c]), errors="coerce")

    boolean = [
        "brake_block", "scanner_violation_flag", "emergency_hold",
        "operator_inside_radius", "series_active"
    ]
    for c in boolean:
        d[c] = bool01(first_existing(d, [c], False))

    categorical = [
        "series_status", "tms_action", "hold_reason", "nav_state",
        "current_segment", "series_run_id", "run_id"
    ]
    for c in categorical:
        d[c] = first_existing(d, [c], "").fillna("").astype(str)

    d["speed"] = (
        d["abs_speed"].where(d["abs_speed"].notna(), d["speed_mps"])
        .fillna(0).clip(lower=0)
    )
    d["soc"] = d["battery_percent"].where(
        d["battery_percent"].notna(), d["battery"]
    )
    d["voltage_v"] = d["cell_voltage_mV"] / 1000.0
    d["session_role"] = role
    d["raw_observation"] = 1
    return d, raw_rows


def gap_aware_1hz(d: pd.DataFrame, max_fill_s: int) -> pd.DataFrame:
    d = d.set_index("timestamp").sort_index()
    numeric_cols = d.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in d.columns if c not in numeric_cols]

    numeric = d[numeric_cols].resample("1s").mean()
    categorical = d[categorical_cols].resample("1s").last()
    observed = d["raw_observation"].resample("1s").max().fillna(0)

    out = pd.concat([numeric, categorical], axis=1)
    out[numeric_cols] = out[numeric_cols].ffill(limit=max_fill_s)
    out[categorical_cols] = out[categorical_cols].ffill(limit=max_fill_s)
    out["is_observed"] = observed.astype(int)

    out = out[out["soc"].notna() & out["speed"].notna()].reset_index()
    out["gap_from_prev_s"] = out["timestamp"].diff().dt.total_seconds().fillna(1)
    out["segment_id"] = (out["gap_from_prev_s"] > max_fill_s + 1).cumsum()
    out["t_s"] = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds()
    return out


def load_graph(nodes_path: Path, edges_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = read_csv(nodes_path).rename(columns={
        "Node": "node", "X-coordinate": "nx", "Y-coordinate": "ny",
        "Node_Degree": "node_degree", "Type_Corridor": "type_corridor",
        "Type_Intersection": "type_intersection", "Type_Station": "type_station"
    })
    edges = read_csv(edges_path).rename(columns={
        "From": "u", "To": "v", "distance": "edge_distance_m"
    })

    for c in ["node", "nx", "ny", "node_degree", "type_corridor",
              "type_intersection", "type_station"]:
        if c in nodes.columns:
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce")
    for c in ["u", "v", "edge_distance_m"]:
        edges[c] = pd.to_numeric(edges[c], errors="coerce")

    edges["edge_id"] = (
        edges["u"].astype("Int64").astype(str) + "->" +
        edges["v"].astype("Int64").astype(str)
    )
    return nodes, edges


def add_context(d: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["snap_node"] = pd.to_numeric(d["snap_node"], errors="coerce").round()
    d["going_to_id"] = pd.to_numeric(d["going_to_id"], errors="coerce").round()
    d["edge_id"] = (
        pd.to_numeric(d["u"], errors="coerce").round().astype("Int64").astype(str)
        + "->" +
        pd.to_numeric(d["v"], errors="coerce").round().astype("Int64").astype(str)
    )

    edge_map = edges.set_index("edge_id")["edge_distance_m"].to_dict()
    d["edge_distance_m"] = d["edge_id"].map(edge_map)
    d["graph_valid_edge"] = d["edge_id"].isin(set(edges["edge_id"])).astype(int)

    node_cols = ["node"] + [
        c for c in ["node_degree", "type_corridor",
                    "type_intersection", "type_station"]
        if c in nodes.columns
    ]
    d = d.merge(
        nodes[node_cols].rename(columns={"node": "snap_node"}),
        on="snap_node", how="left"
    )

    xy = nodes.set_index("node")[["nx", "ny"]]
    d["target_x"] = d["going_to_id"].map(xy["nx"])
    d["target_y"] = d["going_to_id"].map(xy["ny"])
    d["distance_to_target_m"] = np.sqrt(
        (d["x"] - d["target_x"]) ** 2 + (d["y"] - d["target_y"]) ** 2
    )

    run = d["series_run_id"].replace({"nan": "", "None": ""})
    run = run.where(run.str.len() > 0, d["run_id"])
    d["mission_leg_id"] = (
        d["session_role"].astype(str) + "|" + run.astype(str) + "|" +
        d["series_index"].round().astype("Int64").astype(str) + "|" +
        d["series_leg_start_node"].round().astype("Int64").astype(str) + "-" +
        d["series_leg_goal_node"].round().astype("Int64").astype(str)
    )
    return d


BASE_NUM = [
    "soc", "voltage_v", "power_W", "speed", "distance_to_target_m",
    "edge_distance_m", "graph_valid_edge", "node_degree", "type_corridor",
    "type_intersection", "type_station", "hist_speed_mean", "hist_speed_std",
    "hist_stop_share", "hist_power_mean", "hist_power_std", "hist_soc_mean",
    "hist_voltage_mean", "hist_voltage_std", "hist_cmd_mismatch_share",
    "hist_soc_drop", "hist_voltage_drop", "hist_target_progress_m",
    "hist_xy_motion_m", "position_confidence"
]
BASE_CAT = ["edge_id", "snap_node", "going_to_id", "nav_state", "current_segment"]

FEATURE_GROUPS = {
    "soc": ["soc", "voltage_v", "hist_soc_mean", "hist_voltage_mean",
            "hist_voltage_std", "hist_soc_drop", "hist_voltage_drop"],
    "motion": ["speed", "hist_speed_mean", "hist_speed_std",
               "hist_stop_share", "hist_cmd_mismatch_share", "hist_xy_motion_m"],
    "power": ["power_W", "hist_power_mean", "hist_power_std"],
    "progress": ["distance_to_target_m", "hist_target_progress_m"],
    "graph_num": ["edge_distance_m", "graph_valid_edge", "node_degree",
                  "type_corridor", "type_intersection", "type_station",
                  "position_confidence"],
    "graph_cat": BASE_CAT,
}


def add_features_and_labels(
    d: pd.DataFrame, history: int, horizon: int, future_window: int
) -> pd.DataFrame:
    d = d.copy().sort_values(["session_role", "segment_id", "timestamp"])

    status = d["series_status"].fillna("").str.upper()
    action = d["tms_action"].fillna("").str.lower()
    target_valid = d["going_to_id"].notna()
    not_reached = d["target_reached"].fillna(0).lt(0.5)
    active_status = status.eq("SERIES_DEST_SENT")

    d["mission_active"] = (target_valid & not_reached & active_status).astype(int)

    external = (
        status.str.contains("OPERATOR|SCANNER|DEADLOCK|BRAKE_HOLD", regex=True)
        | d["brake_block"].eq(1)
        | d["scanner_violation_flag"].eq(1)
        | d["emergency_hold"].eq(1)
        | d["operator_inside_radius"].eq(1)
        | action.str.contains("hold_|yield_|no_real_goal", regex=True)
    )
    d["external_hold"] = external.astype(int)
    d["primary_eligible"] = (
        d["mission_active"].eq(1) & d["external_hold"].eq(0)
    ).astype(int)

    d["stop"] = d["speed"].lt(0.03).astype(int)
    d["requested_speed"] = d[
        ["requested_straight_speed_mps", "requested_corner_speed_mps"]
    ].max(axis=1)
    d["cmd_motion_mismatch"] = (
        d["requested_speed"].gt(0.05) & d["speed"].lt(0.03)
    ).astype(int)

    group_keys = ["session_role", "segment_id"]
    grp = d.groupby(group_keys, group_keys=False)

    def rolling(col: str, fn: str) -> pd.Series:
        return (
            grp[col].rolling(history, min_periods=max(5, history // 3))
            .agg(fn).reset_index(level=[0, 1], drop=True)
        )

    rolling_specs = [
        ("speed", "mean", "hist_speed_mean"),
        ("speed", "std", "hist_speed_std"),
        ("stop", "mean", "hist_stop_share"),
        ("power_W", "mean", "hist_power_mean"),
        ("power_W", "std", "hist_power_std"),
        ("soc", "mean", "hist_soc_mean"),
        ("voltage_v", "mean", "hist_voltage_mean"),
        ("voltage_v", "std", "hist_voltage_std"),
        ("cmd_motion_mismatch", "mean", "hist_cmd_mismatch_share"),
    ]
    for col, fn, out_col in rolling_specs:
        d[out_col] = rolling(col, fn)

    d["hist_soc_drop"] = -grp["soc"].diff(history)
    d["hist_voltage_drop"] = -grp["voltage_v"].diff(history)
    d["hist_target_progress_m"] = -grp["distance_to_target_m"].diff(history)
    d["hist_xy_motion_m"] = np.sqrt(
        grp["x"].diff(history) ** 2 + grp["y"].diff(history) ** 2
    )

    def future_mean(s: pd.Series) -> pd.Series:
        return (
            s.shift(-horizon).iloc[::-1]
            .rolling(future_window, min_periods=max(3, future_window // 2))
            .mean().iloc[::-1]
        )

    d["future_speed_mean"] = grp["speed"].transform(future_mean)
    d["future_stop_share"] = grp["stop"].transform(future_mean)
    d["future_cmd_mismatch_share"] = grp["cmd_motion_mismatch"].transform(future_mean)
    d["future_active_share"] = grp["mission_active"].transform(future_mean)
    d["future_progress_m"] = grp["distance_to_target_m"].transform(
        lambda s: s.shift(-horizon) - s.shift(-(horizon + future_window - 1))
    )

    d["y_degraded"] = (
        d["primary_eligible"].eq(1)
        & d["future_active_share"].ge(0.6)
        & (
            d["future_cmd_mismatch_share"].ge(0.5)
            | (
                d["future_stop_share"].ge(0.6)
                & d["future_progress_m"].lt(0.03)
            )
            | (
                d["future_speed_mean"].lt(0.055)
                & d["future_progress_m"].lt(0.03)
            )
        )
    ).astype(float)

    invalid = (
        d["primary_eligible"].eq(0)
        | d["future_speed_mean"].isna()
        | d["hist_speed_mean"].isna()
    )
    d.loc[invalid, "y_degraded"] = np.nan
    return d


def feature_set(name: str) -> Tuple[List[str], List[str]]:
    all_num = list(BASE_NUM)
    all_cat = list(BASE_CAT)

    if name == "full":
        return all_num, all_cat
    if name == "no_soc":
        return [x for x in all_num if x not in FEATURE_GROUPS["soc"]], all_cat
    if name == "no_graph":
        return [x for x in all_num if x not in FEATURE_GROUPS["graph_num"]], []
    if name == "no_progress":
        return [x for x in all_num if x not in FEATURE_GROUPS["progress"]], all_cat
    if name == "no_cmd_mismatch":
        return [x for x in all_num if x != "hist_cmd_mismatch_share"], all_cat
    if name == "soc_only":
        return list(FEATURE_GROUPS["soc"]), []
    if name == "motion_only":
        return list(FEATURE_GROUPS["motion"]), []
    if name == "motion_graph":
        return list(dict.fromkeys(
            FEATURE_GROUPS["motion"] + FEATURE_GROUPS["graph_num"]
        )), list(FEATURE_GROUPS["graph_cat"])
    if name == "motion_progress":
        return list(dict.fromkeys(
            FEATURE_GROUPS["motion"] + FEATURE_GROUPS["progress"]
        )), []
    raise ValueError(f"Unknown feature set: {name}")


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())
            ]),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_cols
        ))
    return ColumnTransformer(transformers)


def make_models(num_cols: List[str], cat_cols: List[str]) -> Dict[str, Pipeline]:
    pre = build_preprocessor(num_cols, cat_cols)
    return {
        "logistic": Pipeline([
            ("pre", pre),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
            ))
        ]),
        "extra_trees": Pipeline([
            ("pre", pre),
            ("model", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=3,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "random_forest": Pipeline([
            ("pre", pre),
            ("model", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=3,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
    }


def score_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (p >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "degradation_f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    out["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    out["pr_auc"] = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    return out


def operational_metrics(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = pred.copy()
    d["y_pred"] = (d["y_prob"] >= threshold).astype(int)

    rows = []
    for (protocol, model), g in d.groupby(["protocol", "model"]):
        duration_h = max(
            (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 3600.0,
            1e-9
        )
        mission_stats = []
        for mission, m in g.groupby("mission_leg_id"):
            true_event = int(m["y_true"].max() > 0)
            warned = int(m["y_pred"].max() > 0)
            false_warn = int((true_event == 0) and (warned == 1))

            lead = np.nan
            if true_event and warned:
                first_true = m.loc[m["y_true"].eq(1), "timestamp"].min()
                warnings_before = m.loc[
                    m["y_pred"].eq(1) & m["timestamp"].le(first_true), "timestamp"
                ]
                if len(warnings_before):
                    lead = (first_true - warnings_before.min()).total_seconds()

            mission_stats.append({
                "mission": mission, "true_event": true_event, "warned": warned,
                "false_warn": false_warn, "lead_s": lead
            })

        ms = pd.DataFrame(mission_stats)
        rows.append({
            "protocol": protocol,
            "model": model,
            "threshold": threshold,
            "missions": int(len(ms)),
            "degraded_missions": int(ms["true_event"].sum()),
            "warned_degraded_missions": int(
                ((ms["true_event"] == 1) & (ms["warned"] == 1)).sum()
            ),
            "mission_warning_recall": float(
                ((ms["true_event"] == 1) & (ms["warned"] == 1)).sum()
                / max(ms["true_event"].sum(), 1)
            ),
            "false_warning_missions": int(ms["false_warn"].sum()),
            "false_warnings_per_mission": float(ms["false_warn"].mean()),
            "row_level_false_warnings": int(
                ((g["y_true"] == 0) & (g["y_pred"] == 1)).sum()
            ),
            "false_warning_rows_per_hour": float(
                ((g["y_true"] == 0) & (g["y_pred"] == 1)).sum() / duration_h
            ),
            "median_first_warning_lead_s": float(
                ms["lead_s"].dropna().median()
            ) if ms["lead_s"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate_horizon(
    df: pd.DataFrame, out_dir: Path, horizon: int, threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    low = df[
        df["session_role"].eq("LOW_SOC_STRESS") & df["y_degraded"].notna()
    ].copy()
    high = df[
        df["session_role"].eq("HIGH_SOC_CONTROL") & df["y_degraded"].notna()
    ].copy()

    counts = low["mission_leg_id"].value_counts()
    low = low[low["mission_leg_id"].isin(counts[counts >= 10].index)].copy()
    n_splits = min(5, low["mission_leg_id"].nunique())
    if n_splits < 2:
        raise RuntimeError("Fewer than two usable mission legs.")

    num_cols, cat_cols = feature_set("full")
    rows, predictions = [], []

    for model_name, model in make_models(num_cols, cat_cols).items():
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (tr, te) in enumerate(
            splitter.split(low, low["y_degraded"], low["mission_leg_id"]), 1
        ):
            train, test = low.iloc[tr], low.iloc[te]
            model.fit(train[num_cols + cat_cols], train["y_degraded"].astype(int))
            p = model.predict_proba(test[num_cols + cat_cols])[:, 1]

            row = score_metrics(test["y_degraded"].astype(int).values, p, threshold)
            row.update({
                "horizon_s": horizon, "model": model_name, "fold": fold,
                "protocol": "grouped_low_soc_cv", "feature_set": "full"
            })
            rows.append(row)

            tmp = test[[
                "timestamp", "session_role", "t_s", "mission_leg_id",
                "soc", "speed", "edge_id"
            ]].copy()
            tmp["y_true"] = test["y_degraded"].astype(int).values
            tmp["y_prob"] = p
            tmp["model"] = model_name
            tmp["fold"] = fold
            tmp["protocol"] = "grouped_low_soc_cv"
            tmp["horizon_s"] = horizon
            predictions.append(tmp)

        model.fit(low[num_cols + cat_cols], low["y_degraded"].astype(int))
        if len(high):
            p = model.predict_proba(high[num_cols + cat_cols])[:, 1]
            row = score_metrics(high["y_degraded"].astype(int).values, p, threshold)
            row.update({
                "horizon_s": horizon, "model": model_name, "fold": 0,
                "protocol": "untouched_high_soc_control", "feature_set": "full"
            })
            rows.append(row)

            tmp = high[[
                "timestamp", "session_role", "t_s", "mission_leg_id",
                "soc", "speed", "edge_id"
            ]].copy()
            tmp["y_true"] = high["y_degraded"].astype(int).values
            tmp["y_prob"] = p
            tmp["model"] = model_name
            tmp["fold"] = 0
            tmp["protocol"] = "untouched_high_soc_control"
            tmp["horizon_s"] = horizon
            predictions.append(tmp)

    results = pd.DataFrame(rows)
    pred = pd.concat(predictions, ignore_index=True)
    results.to_csv(out_dir / f"model_results_h{horizon}.csv", index=False)
    pred.to_csv(out_dir / f"predictions_h{horizon}.csv", index=False)
    operational_metrics(pred, threshold).to_csv(
        out_dir / f"operational_metrics_h{horizon}.csv", index=False
    )
    return results, pred


def run_ablation(
    df: pd.DataFrame, out_dir: Path, threshold: float
) -> pd.DataFrame:
    low = df[
        df["session_role"].eq("LOW_SOC_STRESS") & df["y_degraded"].notna()
    ].copy()
    counts = low["mission_leg_id"].value_counts()
    low = low[low["mission_leg_id"].isin(counts[counts >= 10].index)].copy()
    n_splits = min(5, low["mission_leg_id"].nunique())

    settings = [
        "full", "no_soc", "no_graph", "no_progress", "no_cmd_mismatch",
        "soc_only", "motion_only", "motion_graph", "motion_progress"
    ]
    rows = []

    for setting in settings:
        num_cols, cat_cols = feature_set(setting)
        model = make_models(num_cols, cat_cols)["extra_trees"]
        splitter = GroupKFold(n_splits=n_splits)

        for fold, (tr, te) in enumerate(
            splitter.split(low, low["y_degraded"], low["mission_leg_id"]), 1
        ):
            train, test = low.iloc[tr], low.iloc[te]
            model.fit(train[num_cols + cat_cols], train["y_degraded"].astype(int))
            p = model.predict_proba(test[num_cols + cat_cols])[:, 1]
            row = score_metrics(test["y_degraded"].astype(int).values, p, threshold)
            row.update({"feature_set": setting, "fold": fold, "model": "extra_trees"})
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "ablation_h5.csv", index=False)
    return out


def threshold_sensitivity(pred_h5: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    cv = pred_h5[pred_h5["protocol"].eq("grouped_low_soc_cv")]
    for model, g in cv.groupby("model"):
        for threshold in np.arange(0.30, 0.71, 0.05):
            row = score_metrics(
                g["y_true"].astype(int).values,
                g["y_prob"].values,
                float(threshold)
            )
            row.update({"model": model, "threshold": float(threshold)})
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "threshold_sensitivity_h5.csv", index=False)
    return out


def make_plots(
    all_results: pd.DataFrame, ablation: pd.DataFrame,
    threshold_df: pd.DataFrame, out_dir: Path
) -> None:
    summary = (
        all_results[all_results["protocol"].eq("grouped_low_soc_cv")]
        .groupby(["horizon_s", "model"])["macro_f1"]
        .agg(["mean", "std"]).reset_index()
    )
    for model, g in summary.groupby("model"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.errorbar(g["horizon_s"], g["mean"], yerr=g["std"], marker="o", capsize=3)
        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel("Grouped-CV macro-F1")
        ax.set_title(f"Warning horizon performance: {model}")
        ax.set_xticks(sorted(g["horizon_s"].unique()))
        fig.tight_layout()
        fig.savefig(out_dir / f"horizon_{model}.png", dpi=240)
        plt.close(fig)

    a = (
        ablation.groupby("feature_set")["macro_f1"]
        .agg(["mean", "std"]).sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(a.index, a["mean"], yerr=a["std"], capsize=3)
    ax.set_ylabel("Grouped-CV macro-F1")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Feature-group ablation at 5 s")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_h5.png", dpi=240)
    plt.close(fig)

    for model, g in threshold_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(g["threshold"], g["macro_f1"], marker="o", label="Macro-F1")
        ax.plot(g["threshold"], g["recall"], marker="o", label="Recall")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Score")
        ax.set_title(f"Threshold sensitivity: {model}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"threshold_{model}.png", dpi=240)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--out_dir", default="icra_final_results")
    parser.add_argument("--history", type=int, default=30)
    parser.add_argument("--future_window", type=int, default=10)
    parser.add_argument("--max_fill_s", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    low_path = locate(data_dir, ["LOW_SOC", "STRESS"])
    high_path = locate(data_dir, ["HIGH_SOC", "CONTROL"])
    nodes_path = locate(data_dir, ["Node_F3"])
    edges_path = locate(data_dir, ["Edge_Distances3"])

    low_raw, low_raw_rows = load_session(low_path, "LOW_SOC_STRESS")
    high_raw, high_raw_rows = load_session(high_path, "HIGH_SOC_CONTROL")
    low = gap_aware_1hz(low_raw, args.max_fill_s)
    high = gap_aware_1hz(high_raw, args.max_fill_s)
    nodes, edges = load_graph(nodes_path, edges_path)

    base = pd.concat([
        add_context(low, nodes, edges),
        add_context(high, nodes, edges)
    ], ignore_index=True)

    all_results, all_predictions = [], []
    horizon_frames = {}

    for horizon in [0, 5, 10, 20]:
        frame = add_features_and_labels(
            base, args.history, horizon, args.future_window
        )
        frame.to_csv(out_dir / f"processed_h{horizon}.csv", index=False)
        horizon_frames[horizon] = frame

        results, predictions = evaluate_horizon(
            frame, out_dir, horizon, args.threshold
        )
        all_results.append(results)
        all_predictions.append(predictions)

    results = pd.concat(all_results, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    results.to_csv(out_dir / "all_model_results.csv", index=False)
    predictions.to_csv(out_dir / "all_predictions.csv", index=False)

    ablation = run_ablation(horizon_frames[5], out_dir, args.threshold)
    pred_h5 = predictions[predictions["horizon_s"].eq(5)].copy()
    thresholds = threshold_sensitivity(pred_h5, out_dir)
    make_plots(results, ablation, thresholds, out_dir)

# TP / FP / FN case-study figures

    case_pred = pred_h5[
        (pred_h5["model"] == "extra_trees")
        & (pred_h5["protocol"] == "grouped_low_soc_cv")
    ].copy()

    case_pred["timestamp"] = pd.to_datetime(case_pred["timestamp"])
    case_pred["y_pred"] = (
        case_pred["y_prob"] >= args.threshold
    ).astype(int)

    case_context = horizon_frames[5].copy()
    case_context["timestamp"] = pd.to_datetime(
        case_context["timestamp"]
    )

    merge_keys = [
        "timestamp",
        "session_role",
        "t_s",
        "mission_leg_id",
        "soc",
        "speed",
        "edge_id",
    ]

    case_data = case_pred.merge(
        case_context[merge_keys],
        on=merge_keys,
        how="left",
    )

    case_rules = {
        "tp": (
            (case_data["y_true"] == 1)
            & (case_data["y_pred"] == 1)
        ),
        "fp": (
            (case_data["y_true"] == 0)
            & (case_data["y_pred"] == 1)
        ),
        "fn": (
            (case_data["y_true"] == 1)
            & (case_data["y_pred"] == 0)
        ),
    }

    selected_case_rows = []

    for case_name, mask in case_rules.items():

        candidates = case_data[mask].copy()

        if candidates.empty:
            print(f"No {case_name.upper()} case found.")
            continue

        if case_name in ["tp", "fp"]:
            case_row = candidates.sort_values(
                "y_prob",
                ascending=False
            ).iloc[0]
        else:
            case_row = candidates.sort_values(
                "y_prob",
                ascending=True
            ).iloc[0]

        mission_id = case_row["mission_leg_id"]
        centre_time = case_row["timestamp"]

        window = case_data[
            case_data["mission_leg_id"] == mission_id
        ].copy()

        window = window[
            (
                window["timestamp"] >=
                centre_time - pd.Timedelta(seconds=20)
            )
            &
            (
                window["timestamp"] <=
                centre_time + pd.Timedelta(seconds=20)
            )
        ].sort_values("timestamp")

        start_time = window["timestamp"].min()

        window["relative_time_s"] = (
            window["timestamp"] - start_time
        ).dt.total_seconds()

        fig, ax1 = plt.subplots(figsize=(7.2, 4.2))

        ax1.plot(
            window["relative_time_s"],
            window["y_prob"],
            marker="o",
            markersize=3,
            label="Predicted probability",
        )

        ax1.plot(
            window["relative_time_s"],
            window["y_true"],
            linewidth=2,
            label="Degradation label",
        )

        ax1.axhline(
            args.threshold,
            linestyle="--",
            linewidth=1.2,
            label=f"Decision threshold ({args.threshold:.2f})",
        )

        ax1.set_xlabel("Time within displayed window (s)")
        ax1.set_ylabel("Probability / label")
        ax1.set_ylim(-0.05, 1.05)

        ax2 = ax1.twinx()

        ax2.plot(
            window["relative_time_s"],
            window["speed"],
            alpha=0.7,
            label="Speed",
        )

        ax2.set_ylabel("Speed (m/s)")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc="best",
            fontsize=8,
        )

        title_map = {
            "tp": "True-positive warning",
            "fp": "False-positive warning",
            "fn": "Missed degradation",
        }

        ax1.set_title(title_map[case_name])

        fig.tight_layout()
        fig.savefig(
            out_dir / f"case_{case_name}.png",
            dpi=300,
        )
        plt.close(fig)

        selected_case_rows.append({
            "case_type": case_name,
            "timestamp": case_row["timestamp"],
            "mission_leg_id": case_row["mission_leg_id"],
            "y_true": int(case_row["y_true"]),
            "y_pred": int(case_row["y_pred"]),
            "y_prob": float(case_row["y_prob"]),
            "soc": float(case_row["soc"]),
            "speed": float(case_row["speed"]),
            "edge_id": case_row["edge_id"],
        })

    pd.DataFrame(selected_case_rows).to_csv(
        out_dir / "selected_case_windows.csv",
        index=False,
    )

    audit_rows = []
    for role, frame, raw_n in [
        ("LOW_SOC_STRESS", low, low_raw_rows),
        ("HIGH_SOC_CONTROL", high, high_raw_rows)
    ]:
        h5 = horizon_frames[5]
        s = h5[h5["session_role"].eq(role)]
        audit_rows.append({
            "session_role": role,
            "raw_rows": raw_n,
            "gap_aware_aligned_rows": len(frame),
            "observed_share": frame["is_observed"].mean(),
            "active_mission_rows": int(s["mission_active"].sum()),
            "external_hold_rows": int(s["external_hold"].sum()),
            "primary_eligible_rows": int(s["primary_eligible"].sum()),
            "labelled_rows_h5": int(s["y_degraded"].notna().sum()),
            "degraded_rate_h5": float(s["y_degraded"].mean()),
            "mission_legs_h5": int(
                s.loc[s["y_degraded"].notna(), "mission_leg_id"].nunique()
            ),
            "soc_min": float(s["soc"].min()),
            "soc_max": float(s["soc"].max()),
        })
    pd.DataFrame(audit_rows).to_csv(out_dir / "dataset_audit.csv", index=False)

    best = (
        results[results["protocol"].eq("grouped_low_soc_cv")]
        .groupby(["horizon_s", "model"])["macro_f1"]
        .agg(["mean", "std"]).reset_index()
        .sort_values(["horizon_s", "mean"], ascending=[True, False])
        .groupby("horizon_s").head(1)
    )
    report = {
        "raw_input_rows": int(low_raw_rows + high_raw_rows),
        "graph_nodes": int(nodes["node"].nunique()),
        "directed_edges": int(len(edges)),
        "best_by_horizon": best.to_dict(orient="records"),
        
    }
    (out_dir / "run_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
