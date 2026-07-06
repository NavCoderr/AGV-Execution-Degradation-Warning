"""
Real AGV two-session execution-degradation experiment.

Required files in the same folder:
    01_real_agv_live_LOW_SOC_STRESS.csv
    02_real_agv_live_HIGH_SOC_CONTROL.csv
    Node_F3.csv
    Edge_Distances3.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def find_file(data_dir: Path, keys: List[str]) -> Path:
    files = list(data_dir.glob("*.csv"))
    matches = []
    for f in files:
        name = f.name.lower()
        if all(k.lower() in name for k in keys):
            matches.append(f)
    if not matches:
        raise FileNotFoundError(f"No CSV found in {data_dir} with keywords {keys}")
    return sorted(matches, key=lambda p: len(p.name))[0]


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(path, sep=";")
    return df


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def bool01(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)



def load_nodes(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path).rename(columns={
        "Node": "node", "X-coordinate": "x_node", "Y-coordinate": "y_node",
        "Node_Degree": "node_degree", "charging_flag": "charging_flag",
        "Type_Corridor": "type_corridor", "Type_Intersection": "type_intersection",
        "Type_Station": "type_station",
    })
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["node", "x_node", "y_node"]).copy()
    df["node"] = df["node"].astype(int)
    return df


def load_edges(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path).rename(columns={
        "From": "u", "To": "v", "distance": "edge_distance_m",
        "X_from": "x_from", "Y_from": "y_from", "X_to": "x_to", "Y_to": "y_to",
    })
    for c in ["u", "v", "edge_distance_m", "x_from", "y_from", "x_to", "y_to"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["u", "v", "edge_distance_m"]).copy()
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    df["edge_id"] = df["u"].astype(str) + "->" + df["v"].astype(str)
    return df


def load_real(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "agent" in df.columns:
        df = df[(df["agent"].astype(str).str.upper() == "REAL") | df["agent"].isna()].copy()
    if "agent_type" in df.columns:
        ok = df["agent_type"].astype(str).str.lower().isin(["physical", "nan", "none"])
        df = df[ok | df["agent_type"].isna()].copy()
    if "local_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["local_time"], errors="coerce")
    elif "ts" in df.columns:
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit="s")
    else:
        raise ValueError("Real file needs local_time or ts column")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")

    needed = [
        "x", "y", "snap_node", "snap_dist", "speed_mps", "abs_speed", "battery", "battery_percent",
        "power_W", "going_to_id", "target_reached", "brake_block", "brake_release", "position_confidence",
        "scanner_violation_flag", "u", "v", "nav_state", "current_segment"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    for c in ["brake_block", "brake_release", "scanner_violation_flag"]:
        df[c] = bool01(df[c])
    for c in [x for x in needed if x not in ["brake_block", "brake_release", "scanner_violation_flag"]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["speed"] = df["abs_speed"].where(df["abs_speed"].notna(), df["speed_mps"]).fillna(0).clip(lower=0)
    df["battery_percent"] = df["battery_percent"].where(df["battery_percent"].notna(), df["battery"])
    df["power_W"] = df["power_W"].fillna(df["power_W"].median())
    df["target_reached"] = df["target_reached"].fillna(0).clip(0, 1)
    return df


def resample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("timestamp").sort_index()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out = df[numeric_cols].resample("1s").mean().ffill()
    out = out.dropna(how="all")
    out = out.reset_index()
    out["t_s"] = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds()
    for c in ["snap_node", "going_to_id", "u", "v", "target_reached", "brake_block", "scanner_violation_flag"]:
        if c in out.columns:
            out[c] = out[c].round()
    return out



def add_graph_context(df: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    node_xy = nodes.set_index("node")[["x_node", "y_node"]]

    if df["snap_node"].isna().all():
        pts = nodes[["node", "x_node", "y_node"]].to_numpy()
        nearest, dist = [], []
        for x, y in df[["x", "y"]].to_numpy():
            d = np.sqrt((pts[:, 1] - x) ** 2 + (pts[:, 2] - y) ** 2)
            i = int(np.nanargmin(d))
            nearest.append(int(pts[i, 0])); dist.append(float(d[i]))
        df["snap_node"] = nearest
        df["snap_dist"] = dist

    df["snap_node"] = pd.to_numeric(df["snap_node"], errors="coerce").round()
    df["going_to_id"] = pd.to_numeric(df["going_to_id"], errors="coerce").round()

    u = pd.to_numeric(df["u"], errors="coerce").round()
    v = pd.to_numeric(df["v"], errors="coerce").round()
    prev_snap = df["snap_node"].shift(1)
    df["u_edge"] = u.where(u.notna(), prev_snap)
    df["v_edge"] = v.where(v.notna(), df["snap_node"])
    df["edge_id"] = df["u_edge"].astype("Int64").astype(str) + "->" + df["v_edge"].astype("Int64").astype(str)

    valid_edges = set(edges["edge_id"].astype(str))
    df["graph_valid_edge"] = df["edge_id"].isin(valid_edges).astype(int)
    dist_map = edges.set_index("edge_id")["edge_distance_m"].to_dict()
    df["edge_distance_m"] = df["edge_id"].map(dist_map)

    tx = node_xy["x_node"].to_dict(); ty = node_xy["y_node"].to_dict()
    df["target_x"] = df["going_to_id"].map(tx)
    df["target_y"] = df["going_to_id"].map(ty)
    df["distance_to_target_m"] = np.sqrt((df["x"] - df["target_x"])**2 + (df["y"] - df["target_y"])**2)

    ctx_cols = ["node_degree", "charging_flag", "type_corridor", "type_intersection", "type_station"]
    ctx = nodes[["node"] + [c for c in ctx_cols if c in nodes.columns]].copy()
    ctx = ctx.rename(columns={"node": "snap_node"})
    df = df.merge(ctx, on="snap_node", how="left")
    return df



def add_soc_band(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.1, 10, 20, 30, 40, 50, 60, 100]
    labels = [0, 1, 2, 3, 4, 5, 6]
    df["soc_band_code"] = pd.cut(df["battery_percent"], bins=bins, labels=labels, include_lowest=True).astype(float)
    band_names = {0:"0-10", 1:"10-20", 2:"20-30", 3:"30-40", 4:"40-50", 5:"50-60", 6:"60+"}
    df["soc_band"] = df["soc_band_code"].map(band_names)
    return df


def first_minus_last(x: np.ndarray) -> float:
    if len(x) == 0 or np.isnan(x[0]) or np.isnan(x[-1]):
        return np.nan
    return float(x[0] - x[-1])


def make_features(df: pd.DataFrame, history_window: int) -> pd.DataFrame:
    df = df.copy()
    w = history_window
    df["stop_flag"] = (df["speed"] < 0.03).astype(int)
    df["very_slow_flag"] = (df["speed"] < 0.05).astype(int)
    df["moving_flag"] = (df["speed"] >= 0.03).astype(int)
    df["low_soc_20"] = (df["battery_percent"] < 20).astype(int)
    df["low_soc_30"] = (df["battery_percent"] < 30).astype(int)
    df["critical_soc_10"] = (df["battery_percent"] < 10).astype(int)

    r = lambda s, f: s.rolling(w, min_periods=max(3, w//3)).agg(f)
    df["hist_speed_mean"] = r(df["speed"], "mean")
    df["hist_speed_std"] = r(df["speed"], "std")
    df["hist_speed_min"] = r(df["speed"], "min")
    df["hist_speed_max"] = r(df["speed"], "max")
    df["hist_stop_share"] = r(df["stop_flag"], "mean")
    df["hist_very_slow_share"] = r(df["very_slow_flag"], "mean")
    df["hist_moving_share"] = r(df["moving_flag"], "mean")
    df["hist_power_mean"] = r(df["power_W"], "mean")
    df["hist_power_std"] = r(df["power_W"], "std")
    df["hist_soc_mean"] = r(df["battery_percent"], "mean")
    df["hist_soc_min"] = r(df["battery_percent"], "min")
    df["hist_soc_drop"] = df["battery_percent"].rolling(w, min_periods=max(3, w//3)).apply(first_minus_last, raw=True)
    df["hist_brake_share"] = r(df["brake_block"], "mean")
    df["hist_scanner_violation_share"] = r(df["scanner_violation_flag"], "mean")
    df["hist_target_progress_m"] = df["distance_to_target_m"].rolling(w, min_periods=max(3, w//3)).apply(first_minus_last, raw=True)
    df["hist_xy_motion_m"] = np.sqrt((df["x"] - df["x"].shift(w))**2 + (df["y"] - df["y"].shift(w))**2)

    df["current_speed"] = df["speed"]
    df["current_soc"] = df["battery_percent"]
    df["current_power_W"] = df["power_W"]
    df["current_distance_to_target_m"] = df["distance_to_target_m"]
    df["current_edge_distance_m"] = df["edge_distance_m"]
    df["current_graph_valid_edge"] = df["graph_valid_edge"]
    return df


def future_mean(s: pd.Series, start: int, window: int) -> pd.Series:
    arr = s.to_numpy(dtype=float)
    n = len(arr); out = np.full(n, np.nan)
    for i in range(n):
        a = i + start; b = min(n, a + window)
        if a < n and b > a:
            out[i] = np.nanmean(arr[a:b])
    return pd.Series(out, index=s.index)


def future_progress(dist: pd.Series, start: int, window: int) -> pd.Series:
    arr = dist.to_numpy(dtype=float)
    n = len(arr); out = np.full(n, np.nan)
    for i in range(n):
        a = i + start; b = min(n, a + window)
        if a < n and b > a and not np.isnan(arr[a]) and not np.isnan(arr[b-1]):
            out[i] = arr[a] - arr[b-1]
    return pd.Series(out, index=dist.index)


def add_degradation_label(df: pd.DataFrame, horizon: int, future_window: int) -> pd.DataFrame:
    df = df.copy()
    h = horizon
    df[f"future_speed_mean_h{h}"] = future_mean(df["speed"], h, future_window)
    df[f"future_stop_share_h{h}"] = future_mean(df["stop_flag"], h, future_window)
    df[f"future_brake_share_h{h}"] = future_mean(df["brake_block"], h, future_window)
    df[f"future_progress_m_h{h}"] = future_progress(df["distance_to_target_m"], h, future_window)

    not_reached = df["target_reached"].fillna(0) < 0.5
    degraded = (
        (df[f"future_speed_mean_h{h}"] < 0.055) |
        (df[f"future_stop_share_h{h}"] >= 0.60) |
        ((df[f"future_progress_m_h{h}"] < 0.03) & not_reached) |
        (df[f"future_brake_share_h{h}"] >= 0.30)
    )
    df[f"degraded_h{h}"] = degraded.astype(float)
    req = [f"future_speed_mean_h{h}", f"future_stop_share_h{h}", f"future_progress_m_h{h}"]
    df.loc[df[req].isna().any(axis=1), f"degraded_h{h}"] = np.nan
    return df



def make_soc_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("soc_band", observed=False)
    return g.agg(
        rows=("speed", "size"),
        mean_speed_mps=("speed", "mean"),
        median_speed_mps=("speed", "median"),
        stop_share=("stop_flag", "mean"),
        very_slow_share=("very_slow_flag", "mean"),
        mean_power_W=("power_W", "mean"),
        target_reached_share=("target_reached", "mean"),
        graph_valid_edge_share=("graph_valid_edge", "mean"),
    ).reset_index()


def make_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[df["graph_valid_edge"] == 1].copy()
    if tmp.empty:
        return pd.DataFrame()
    tmp["low_soc_20"] = (tmp["battery_percent"] < 20).astype(int)
    g = tmp.groupby("edge_id")
    return g.agg(
        rows=("speed", "size"),
        mean_speed_mps=("speed", "mean"),
        stop_share=("stop_flag", "mean"),
        min_soc=("battery_percent", "min"),
        mean_soc=("battery_percent", "mean"),
        low_soc_20_share=("low_soc_20", "mean"),
        mean_power_W=("power_W", "mean"),
        edge_distance_m=("edge_distance_m", "median"),
    ).reset_index().sort_values(["low_soc_20_share", "stop_share", "rows"], ascending=False)


def save_plots(df: pd.DataFrame, soc: pd.DataFrame, out_dir: Path) -> None:
    p = out_dir / "plots"; ensure_dir(p)
    plt.figure(figsize=(10,4)); plt.plot(df["timestamp"], df["battery_percent"])
    plt.xlabel("Time"); plt.ylabel("Battery/SOC (%)"); plt.title("Real AGV SOC over time")
    plt.tight_layout(); plt.savefig(p/"soc_over_time.png", dpi=200); plt.close()

    plt.figure(figsize=(10,4)); plt.plot(df["timestamp"], df["speed"])
    plt.xlabel("Time"); plt.ylabel("Speed (m/s)"); plt.title("Real AGV speed over time")
    plt.tight_layout(); plt.savefig(p/"speed_over_time.png", dpi=200); plt.close()

    if not soc.empty:
        plt.figure(figsize=(8,4)); plt.bar(soc["soc_band"].astype(str), soc["stop_share"])
        plt.xlabel("SOC band (%)"); plt.ylabel("Stop share"); plt.title("Stop share by SOC band")
        plt.tight_layout(); plt.savefig(p/"stop_share_by_soc_band.png", dpi=200); plt.close()

        plt.figure(figsize=(8,4)); plt.bar(soc["soc_band"].astype(str), soc["mean_speed_mps"])
        plt.xlabel("SOC band (%)"); plt.ylabel("Mean speed (m/s)"); plt.title("Mean speed by SOC band")
        plt.tight_layout(); plt.savefig(p/"mean_speed_by_soc_band.png", dpi=200); plt.close()



FEATURES = [
    "current_speed", "current_soc", "current_power_W", "current_distance_to_target_m",
    "current_edge_distance_m", "current_graph_valid_edge", "snap_dist", "position_confidence",
    "soc_band_code", "snap_node", "going_to_id", "node_degree", "charging_flag",
    "type_corridor", "type_intersection", "type_station", "low_soc_20", "low_soc_30",
    "critical_soc_10", "hist_speed_mean", "hist_speed_std", "hist_speed_min", "hist_speed_max",
    "hist_stop_share", "hist_very_slow_share", "hist_moving_share", "hist_power_mean", "hist_power_std",
    "hist_soc_mean", "hist_soc_min", "hist_soc_drop", "hist_brake_share", "hist_scanner_violation_share",
    "hist_target_progress_m", "hist_xy_motion_m", "edge_distance_m", "distance_to_target_m",
    "nav_state", "current_segment"
]
SOC_FEATURES = ["current_soc", "soc_band_code", "low_soc_20", "low_soc_30", "critical_soc_10", "hist_soc_mean", "hist_soc_min", "hist_soc_drop"]
MOTION_FEATURES = ["current_speed", "hist_speed_mean", "hist_speed_std", "hist_speed_min", "hist_speed_max", "hist_stop_share", "hist_very_slow_share", "hist_moving_share", "hist_xy_motion_m"]
POWER_FEATURES = ["current_power_W", "hist_power_mean", "hist_power_std"]
GRAPH_FEATURES = ["current_edge_distance_m", "current_graph_valid_edge", "snap_dist", "snap_node", "going_to_id", "node_degree", "charging_flag", "type_corridor", "type_intersection", "type_station", "edge_distance_m", "nav_state", "current_segment"]
TARGET_FEATURES = ["current_distance_to_target_m", "distance_to_target_m", "hist_target_progress_m"]
SAFETY_FEATURES = ["hist_brake_share", "hist_scanner_violation_share", "position_confidence"]
FEATURE_GROUPS = {
    "FULL": FEATURES,
    "NO_SOC": MOTION_FEATURES + POWER_FEATURES + GRAPH_FEATURES + TARGET_FEATURES + SAFETY_FEATURES,
    "NO_GRAPH": SOC_FEATURES + MOTION_FEATURES + POWER_FEATURES + TARGET_FEATURES + SAFETY_FEATURES,
    "NO_TARGET_PROGRESS": SOC_FEATURES + MOTION_FEATURES + POWER_FEATURES + GRAPH_FEATURES + SAFETY_FEATURES,
    "ONLY_SOC": SOC_FEATURES,
    "ONLY_MOTION": MOTION_FEATURES,
    "SOC_MOTION": SOC_FEATURES + MOTION_FEATURES,
    "MOTION_GRAPH": MOTION_FEATURES + GRAPH_FEATURES,
    "SOC_MOTION_GRAPH": SOC_FEATURES + MOTION_FEATURES + GRAPH_FEATURES,
}
LABEL_CONFIGS = {
    "loose": {"speed_thr": 0.080, "stop_thr": 0.50, "progress_thr": 0.05, "brake_thr": 0.20},
    "default": {"speed_thr": 0.055, "stop_thr": 0.60, "progress_thr": 0.03, "brake_thr": 0.30},
    "strict": {"speed_thr": 0.040, "stop_thr": 0.70, "progress_thr": 0.01, "brake_thr": 0.40},
}


def add_degradation_label_config(df: pd.DataFrame, horizon: int, future_window: int, config_name: str, cfg: Dict[str, float]) -> pd.DataFrame:
    df = df.copy(); h = horizon
    if f"future_speed_mean_h{h}" not in df.columns:
        df[f"future_speed_mean_h{h}"] = future_mean(df["speed"], h, future_window)
        df[f"future_stop_share_h{h}"] = future_mean(df["stop_flag"], h, future_window)
        df[f"future_brake_share_h{h}"] = future_mean(df["brake_block"], h, future_window)
        df[f"future_progress_m_h{h}"] = future_progress(df["distance_to_target_m"], h, future_window)
    not_reached = df["target_reached"].fillna(0) < 0.5
    label = f"degraded_{config_name}_h{h}"
    degraded = (
        (df[f"future_speed_mean_h{h}"] < cfg["speed_thr"]) |
        (df[f"future_stop_share_h{h}"] >= cfg["stop_thr"]) |
        ((df[f"future_progress_m_h{h}"] < cfg["progress_thr"]) & not_reached) |
        (df[f"future_brake_share_h{h}"] >= cfg["brake_thr"])
    )
    df[label] = degraded.astype(float)
    req=[f"future_speed_mean_h{h}", f"future_stop_share_h{h}", f"future_progress_m_h{h}"]
    df.loc[df[req].isna().any(axis=1), label] = np.nan
    return df


def get_models(kind="all"):
    models = {
        "LogisticRegression": Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1200, class_weight="balanced"))]),
        "ExtraTrees": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", ExtraTreesClassifier(n_estimators=25, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1))]),
    }
    if kind == "all":
        models.update({
            "RandomForest": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=25, random_state=42, class_weight="balanced", min_samples_leaf=2, n_jobs=-1))]),
            "GradientBoosting": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", GradientBoostingClassifier(random_state=42, n_estimators=25, max_depth=3))]),
        })
    if kind == "extra":
        return {"ExtraTrees": models["ExtraTrees"]}
    return models


def metrics_row(y_true, y_pred, y_prob):
    out={
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "degradation_precision": precision_score(y_true, y_pred, zero_division=0),
        "degradation_recall": recall_score(y_true, y_pred, zero_division=0),
        "degradation_f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2 and y_prob is not None:
        out.update({"roc_auc": roc_auc_score(y_true, y_prob), "pr_auc": average_precision_score(y_true, y_prob), "brier": brier_score_loss(y_true, np.clip(y_prob,0,1))})
    else:
        out.update({"roc_auc":np.nan,"pr_auc":np.nan,"brier":np.nan})
    try: tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    except Exception: tn=fp=fn=tp=0
    out.update({"tn":tn,"fp":fp,"fn":fn,"tp":tp})
    return out


def blocked_evaluate(data, label, features, models, meta, train_frac=0.70):
    rows=[]; preds=[]
    data = data.dropna(subset=[label]).dropna(subset=["hist_speed_mean","hist_stop_share","hist_soc_mean"]).copy()
    data.loc[:, label] = data[label].astype(int)
    counts=data[label].value_counts().to_dict()
    if len(counts)<2 or min(counts.values())<10 or len(data)<80:
        row=dict(meta); row.update({"model":"SKIPPED","status":"too_few_samples_or_one_class","rows":len(data),"class_counts":json.dumps(counts)})
        return [row], preds
    features=[f for f in features if f in data.columns]
    split=int(len(data)*train_frac)
    Xtr=data.iloc[:split][features]; Xte=data.iloc[split:][features]
    ytr=data.iloc[:split][label].to_numpy(); yte=data.iloc[split:][label].to_numpy()
    if len(np.unique(ytr))<2 or len(np.unique(yte))<2:
        row=dict(meta); row.update({"model":"SKIPPED","status":"one_class_in_time_split","rows":len(data),"class_counts":json.dumps(counts)})
        return [row], preds
    for name, model in models.items():
        model.fit(Xtr,ytr)
        yp=model.predict(Xte)
        prob=model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        m=metrics_row(yte,yp,prob); m.update(dict(meta))
        m.update({"model":name,"status":"ok","train_rows":len(Xtr),"test_rows":len(Xte),"positive_rate_test":float(np.mean(yte)),"features_used":len(features)})
        rows.append(m)
        if meta.get("feature_group")=="FULL" and meta.get("label_config")=="default":
            pr=data.iloc[split:][["timestamp","t_s","battery_percent","speed","edge_id","snap_node","going_to_id",label]].copy()
            pr["horizon_s"]=meta.get("horizon_s"); pr["model"]=name; pr["y_true"]=yte; pr["y_pred"]=yp; pr["y_prob"]=prob if prob is not None else np.nan
            preds.append(pr)
    return rows,preds


def run_experiments(df, horizons, future_window, out_dir):
    all_rows=[]; all_preds=[]; all_importance=[]
    for h in horizons:
        for cname,cfg in LABEL_CONFIGS.items():
            df=add_degradation_label_config(df,h,future_window,cname,cfg)
        label=f"degraded_default_h{h}"
        dataset=df.dropna(subset=[label]).dropna(subset=["hist_speed_mean","hist_stop_share","hist_soc_mean"]).copy()
        dataset[label]=dataset[label].astype(int)
        dataset.to_csv(out_dir/f"dataset_horizon_{h}s.csv", index=False)

    for h in horizons:
        rows,preds=blocked_evaluate(df, f"degraded_default_h{h}", FEATURE_GROUPS["FULL"], get_models("all"), {"experiment":"model_comparison","feature_group":"FULL","label_config":"default","horizon_s":h})
        all_rows+=rows; all_preds+=preds

    for h in horizons:
        for gname,gfeatures in FEATURE_GROUPS.items():
            rows,_=blocked_evaluate(df, f"degraded_default_h{h}", gfeatures, get_models("extra"), {"experiment":"feature_ablation","feature_group":gname,"label_config":"default","horizon_s":h})
            all_rows+=rows

    for h in horizons:
        for cname in LABEL_CONFIGS:
            rows,_=blocked_evaluate(df, f"degraded_{cname}_h{h}", FEATURE_GROUPS["FULL"], get_models("extra"), {"experiment":"label_sensitivity","feature_group":"FULL","label_config":cname,"horizon_s":h})
            all_rows+=rows

    for h in horizons:
        label=f"degraded_default_h{h}"
        data=df.dropna(subset=[label]).dropna(subset=["hist_speed_mean","hist_stop_share","hist_soc_mean"]).copy()
        if len(data)>80 and data[label].nunique()==2:
            data[label]=data[label].astype(int)
            cols=[c for c in FEATURE_GROUPS["FULL"] if c in data.columns]
            X=data[cols]; y=data[label].to_numpy()
            et=Pipeline([("imputer",SimpleImputer(strategy="median")),("clf",ExtraTreesClassifier(n_estimators=30,random_state=42,class_weight="balanced",min_samples_leaf=2,n_jobs=-1))])
            et.fit(X,y)
            all_importance.append(pd.DataFrame({"horizon_s":h,"feature":cols,"importance":et.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False))

    metrics=pd.DataFrame(all_rows)
    metrics.to_csv(out_dir/"degradation_metrics_all_experiments.csv",index=False)
    metrics.to_csv(out_dir/"degradation_metrics_blocked_split.csv",index=False)
    if all_preds:
        pd.concat(all_preds,ignore_index=True).to_csv(out_dir/"degradation_predictions_full_blocked_split.csv",index=False)
    if all_importance:
        pd.concat(all_importance,ignore_index=True).to_csv(out_dir/"feature_importance.csv",index=False)
    def agg(sub, group):
        sub=sub[sub["status"].eq("ok")].copy()
        if sub.empty: return pd.DataFrame()
        return sub.groupby(group,as_index=False).agg(
            rows=("test_rows","mean"), accuracy=("accuracy","mean"), macro_f1=("macro_f1","mean"),
            degradation_recall=("degradation_recall","mean"), degradation_f1=("degradation_f1","mean"),
            roc_auc=("roc_auc","mean"), pr_auc=("pr_auc","mean"), brier=("brier","mean"), features_used=("features_used","mean")
        ).sort_values(["macro_f1","degradation_recall"],ascending=False)
    model_table=agg(metrics[metrics["experiment"].eq("model_comparison")],["horizon_s","model"])
    ablation_table=agg(metrics[metrics["experiment"].eq("feature_ablation")],["horizon_s","feature_group","model"])
    sensitivity_table=agg(metrics[metrics["experiment"].eq("label_sensitivity")],["horizon_s","label_config","model"])
    model_table.to_csv(out_dir/"model_comparison.csv",index=False)
    ablation_table.to_csv(out_dir/"feature_ablation.csv",index=False)
    sensitivity_table.to_csv(out_dir/"label_sensitivity.csv",index=False)
    model_table.to_csv(out_dir/"degradation_metrics_aggregated.csv",index=False)
    rows=[]
    for name, table in [("model_comparison",model_table),("feature_ablation",ablation_table),("label_sensitivity",sensitivity_table)]:
        if not table.empty:
            r=table.iloc[0].to_dict(); r["table"]=name; rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir/"best_results_summary.csv",index=False)


def save_advanced_plots(out_dir: Path):
    p=out_dir/"plots"; ensure_dir(p)
    model_path=out_dir/"model_comparison.csv"; ab_path=out_dir/"feature_ablation.csv"
    if model_path.exists():
        m=pd.read_csv(model_path)
        if not m.empty:
            best=m.sort_values(["horizon_s","macro_f1"],ascending=[True,False]).groupby("horizon_s").head(1)
            plt.figure(figsize=(7,4)); plt.plot(best["horizon_s"], best["macro_f1"], marker="o")
            plt.xlabel("Prediction horizon (s)"); plt.ylabel("Best macro-F1"); plt.title("Early-warning performance vs horizon")
            plt.tight_layout(); plt.savefig(p/"best_macro_f1_by_horizon.png",dpi=200); plt.close()
    if ab_path.exists():
        ab=pd.read_csv(ab_path)
        cand=ab[(ab["horizon_s"]==5)&(ab["model"]=="ExtraTrees")]
        if cand.empty: cand=ab[ab["model"]=="ExtraTrees"]
        if not cand.empty:
            cand=cand.sort_values("macro_f1",ascending=False)
            plt.figure(figsize=(10,4)); plt.bar(cand["feature_group"].astype(str), cand["macro_f1"])
            plt.xticks(rotation=35,ha="right"); plt.ylabel("Macro-F1"); plt.title("Feature-group ablation")
            plt.tight_layout(); plt.savefig(p/"feature_ablation_macro_f1.png",dpi=200); plt.close()




def find_files(data_dir: Path, keys: List[str]) -> List[Path]:
    files = list(data_dir.glob("*.csv"))
    matches = []
    for f in files:
        name = f.name.lower()
        if all(k.lower() in name for k in keys):
            matches.append(f)
    return sorted(matches, key=lambda p: p.name)


def classify_session(raw: pd.DataFrame) -> str:
    soc_min = float(pd.to_numeric(raw.get("battery_percent", pd.Series(dtype=float)), errors="coerce").min())
    soc_max = float(pd.to_numeric(raw.get("battery_percent", pd.Series(dtype=float)), errors="coerce").max())
    if np.isfinite(soc_min) and soc_min < 25:
        return "LOW_SOC_STRESS"
    if np.isfinite(soc_min) and soc_min >= 60:
        return "HIGH_SOC_CONTROL"
    return "MID_SOC_SESSION"


def session_summary(df: pd.DataFrame, raw_stats: List[Dict]) -> pd.DataFrame:
    rows=[]
    for sid, g in df.groupby("session_id"):
        rows.append({
            "session_id": sid,
            "session_role": g["session_role"].iloc[0],
            "rows_1hz": int(len(g)),
            "battery_min_percent": float(g["battery_percent"].min()),
            "battery_max_percent": float(g["battery_percent"].max()),
            "mean_speed_mps": float(g["speed"].mean()),
            "stop_share": float((g["speed"] < 0.03).mean()),
            "very_slow_share": float((g["speed"] < 0.05).mean()),
            "mean_power_W": float(g["power_W"].mean()),
            "graph_valid_edge_share": float(g["graph_valid_edge"].mean()),
            "degraded_h5_rate_default": float(g.get("degraded_default_h5", pd.Series(index=g.index, dtype=float)).mean()) if "degraded_default_h5" in g.columns else np.nan,
        })
    out=pd.DataFrame(rows)
    raw=pd.DataFrame(raw_stats)
    return out.merge(raw, on=["session_id","session_role"], how="left") if not raw.empty else out


def save_probability_timeline_plot(pred: pd.DataFrame, out_dir: Path, session_role: str, horizon: int = 5) -> None:
    p = out_dir / "plots"; ensure_dir(p)
    sub = pred[(pred["session_role"] == session_role) & (pred["horizon_s"] == horizon)].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["session_id", "t_s"]).head(1200)
    fig, ax1 = plt.subplots(figsize=(12,4))
    ax1.plot(sub["t_s"], sub["y_prob"], label="Predicted degradation probability")
    ax1.plot(sub["t_s"], sub["y_true"], label="True degraded label", alpha=0.7)
    ax1.axhline(0.4, linestyle="--", linewidth=1, label="warning threshold 0.40")
    ax1.axhline(0.7, linestyle=":", linewidth=1, label="high-risk threshold 0.70")
    ax1.set_xlabel("Time within session (s)")
    ax1.set_ylabel("Probability / label")
    ax2 = ax1.twinx()
    ax2.plot(sub["t_s"], sub["battery_percent"], alpha=0.35, label="SOC (%)")
    ax2.set_ylabel("SOC (%)")
    ax1.set_title(f"Warning probability timeline: {session_role}, horizon={horizon}s")
    lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
    fig.tight_layout(); fig.savefig(p/f"warning_probability_timeline_{session_role}_h{horizon}.png", dpi=200); plt.close(fig)


def run_session_control_advisory(df: pd.DataFrame, out_dir: Path, horizon: int = 5) -> None:
    """Train a practical full-feature GradientBoosting warning model on the low-SOC session and apply it to the high-SOC control.
    This is not a safety controller. It produces advisory categories: continue, warning, charge_check.
    """
    label = f"degraded_default_h{horizon}"
    if label not in df.columns:
        df = add_degradation_label_config(df, horizon, 10, "default", LABEL_CONFIGS["default"])
    cols = [c for c in FEATURE_GROUPS["FULL"] if c in df.columns]
    data = df.dropna(subset=[label]).dropna(subset=["hist_speed_mean","hist_stop_share","hist_soc_mean"]).copy()
    if data.empty or data[label].nunique() < 2:
        return
    data[label] = data[label].astype(int)
    low = data[data["session_role"] == "LOW_SOC_STRESS"].copy()
    high = data[data["session_role"] == "HIGH_SOC_CONTROL"].copy()
    if len(low) < 100 or low[label].nunique() < 2:
        train = data.iloc[:int(len(data)*0.7)].copy()
        test = data.iloc[int(len(data)*0.7):].copy()
    else:
        split = int(len(low)*0.7)
        train = low.iloc[:split].copy()
        test = pd.concat([low.iloc[split:].copy(), high.copy()], ignore_index=True)
    if train[label].nunique() < 2 or test.empty:
        return
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", GradientBoostingClassifier(random_state=42, n_estimators=30, max_depth=3))])
    model.fit(train[cols], train[label].to_numpy())
    prob = model.predict_proba(test[cols])[:,1]
    pred = (prob >= 0.5).astype(int)
    out = test[["session_id","session_role","timestamp","t_s","battery_percent","speed","edge_id","snap_node","going_to_id",label]].copy()
    out["horizon_s"] = horizon
    out["y_true"] = test[label].to_numpy()
    out["y_prob"] = prob
    out["y_pred"] = pred
    out["advisory"] = np.where((out["y_prob"] >= 0.70) & (out["battery_percent"] < 15), "charge_check_or_safe_node",
                        np.where(out["y_prob"] >= 0.40, "operator_warning", "continue"))
    out.to_csv(out_dir/"advisory_predictions_low_train_high_control_h5.csv", index=False)
    summary = out.groupby(["session_role"], as_index=False).agg(
        rows=("y_prob","size"),
        true_degraded_rate=("y_true","mean"),
        mean_warning_probability=("y_prob","mean"),
        warning_share_prob_ge_040=("y_prob", lambda x: float((x>=0.40).mean())),
        high_risk_share_prob_ge_070=("y_prob", lambda x: float((x>=0.70).mean())),
        charge_check_share=("advisory", lambda x: float((x=="charge_check_or_safe_node").mean())),
    )
    summary.to_csv(out_dir/"session_control_advisory_summary.csv", index=False)
    save_probability_timeline_plot(out, out_dir, "LOW_SOC_STRESS", horizon)
    save_probability_timeline_plot(out, out_dir, "HIGH_SOC_CONTROL", horizon)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out_dir", default="low_soc_real_two_session_results")
    ap.add_argument("--history_window", type=int, default=30)
    ap.add_argument("--future_window", type=int, default=10)
    ap.add_argument("--horizons", default="0,5,10,20")
    args=ap.parse_args()
    data_dir=Path(args.data_dir).resolve(); out_dir=Path(args.out_dir).resolve(); ensure_dir(out_dir)
    real_files=find_files(data_dir,["real","agv","live"])
    if not real_files:
        raise FileNotFoundError(f"No REAL AGV live CSV found in {data_dir}. Put the two real AGV live CSV files in this folder.")
    node_path=find_file(data_dir,["node"]); edge_path=find_file(data_dir,["edge","distance"])
    nodes=load_nodes(node_path); edges=load_edges(edge_path)
    all_dfs=[]; raw_stats=[]
    for i, real_path in enumerate(real_files, start=1):
        raw=load_real(real_path)
        role=classify_session(raw)
        sid=f"S{i}_{role}"
        d=resample_1hz(raw)
        d=add_graph_context(d,nodes,edges)
        d=add_soc_band(d)
        d=make_features(d,args.history_window)
        d["session_id"]=sid
        d["session_role"]=role
        d["source_file"]=real_path.name
        all_dfs.append(d)
        raw_stats.append({
            "session_id":sid,"session_role":role,"source_file":real_path.name,
            "raw_rows":int(len(raw)),
            "raw_battery_min_percent":float(raw["battery_percent"].min()),
            "raw_battery_max_percent":float(raw["battery_percent"].max()),
        })
    df=pd.concat(all_dfs, ignore_index=True)
    horizons=[int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    for h in horizons:
        df=add_degradation_label(df,h,args.future_window)
        for cname,cfg in LABEL_CONFIGS.items():
            df=add_degradation_label_config(df,h,args.future_window,cname,cfg)
    df.to_csv(out_dir/"processed_real_agv_1hz_features_labels_two_sessions.csv",index=False)
    sess = session_summary(df, raw_stats); sess.to_csv(out_dir/"session_summary_low_vs_high_soc.csv", index=False)
    soc=df.groupby(["session_role","soc_band"], observed=False).agg(
        rows=("speed","size"), mean_speed_mps=("speed","mean"), median_speed_mps=("speed","median"),
        stop_share=("stop_flag","mean"), very_slow_share=("very_slow_flag","mean"), mean_power_W=("power_W","mean"),
        degraded_h5_rate=("degraded_default_h5","mean") if "degraded_default_h5" in df.columns else ("speed","mean")
    ).reset_index()
    soc.to_csv(out_dir/"soc_band_by_session_summary.csv",index=False)
    make_soc_summary(df).to_csv(out_dir/"soc_band_real_agv_summary.csv",index=False)
    make_edge_summary(df).to_csv(out_dir/"edge_context_real_agv_summary.csv",index=False)
    save_plots(df, make_soc_summary(df), out_dir)
    run_experiments(df,horizons,args.future_window,out_dir)
    run_session_control_advisory(df,out_dir,horizon=5)
    save_advanced_plots(out_dir)
    summary={
        "experiment_type":"real_agv_low_soc_execution_degradation_with_high_soc_control",
        "real_files":[p.name for p in real_files],
        "node_file":node_path.name,"edge_file":edge_path.name,
        "sessions":raw_stats,
        "total_processed_1hz_rows":int(len(df)),"graph_nodes":int(len(nodes)),"graph_edges":int(len(edges)),
        "battery_min_percent":float(df["battery_percent"].min()),"battery_max_percent":float(df["battery_percent"].max()),
        "history_window_s":args.history_window,"future_window_s":args.future_window,"horizons_s":horizons,
        "models":list(get_models("all").keys()),"feature_ablation_groups":list(FEATURE_GROUPS.keys()),"label_sensitivity_configs":LABEL_CONFIGS,
        "main_claim":"Learned early warning of degraded AGV execution from real physical telemetry, SOC dynamics, motion stability, target progress, graph context, and a high-SOC control session.",
        "not_claimed":["no SIM data","no edge-cost correction","no dynamic replanning","no calibrated battery-health model","no certified safety controller"]
    }
    (out_dir/"experiment_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("DONE")
    print(f"Output folder: {out_dir}")
    print(f"Real files used: {', '.join([p.name for p in real_files])}")
    print(f"Rows: 1Hz={len(df)}, nodes={len(nodes)}, edges={len(edges)}")
    print(f"Battery range: {df['battery_percent'].min():.1f}% to {df['battery_percent'].max():.1f}%")
    print("Main tables: session_summary_low_vs_high_soc.csv, session_control_advisory_summary.csv, model_comparison.csv")

if __name__=="__main__":
    main()
