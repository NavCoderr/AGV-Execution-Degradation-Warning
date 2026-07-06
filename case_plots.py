#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_window_id(df: pd.DataFrame, window_s: int) -> pd.DataFrame:
    out = df.copy()
    out["window_start_s"] = (np.floor(out["t_s"] / window_s) * window_s).astype(int)
    out["window_end_s"] = out["window_start_s"] + window_s
    return out


def make_window_table(pred: pd.DataFrame, window_s: int) -> pd.DataFrame:
    pred = add_window_id(pred, window_s)
    g = pred.groupby(["session_role", "window_start_s", "window_end_s"], as_index=False)
    w = g.agg(
        rows=("y_prob", "size"),
        mean_prob=("y_prob", "mean"),
        max_prob=("y_prob", "max"),
        warning_share=("y_prob", lambda x: float((x >= 0.40).mean())),
        high_risk_share=("y_prob", lambda x: float((x >= 0.70).mean())),
        rule_degraded_rate=("y_true", "mean"),
        mean_soc=("battery_percent", "mean"),
        min_soc=("battery_percent", "min"),
        max_soc=("battery_percent", "max"),
        mean_speed=("speed", "mean"),
        stop_share=("speed", lambda x: float((x < 0.03).mean())),
        very_slow_share=("speed", lambda x: float((x < 0.05).mean())),
    )
    w = w[w["rows"] >= max(5, int(window_s * 0.5))].copy()
    return w


def pick_candidates(w: pd.DataFrame, per_group: int) -> pd.DataFrame:
    parts = []
    low = w[w["session_role"].eq("LOW_SOC_STRESS")].copy()
    high = w[w["session_role"].eq("HIGH_SOC_CONTROL")].copy()

    a = low.sort_values(["mean_prob", "rule_degraded_rate", "stop_share"], ascending=False).head(per_group)
    a["case_type"] = "LOW_SOC_high_warning_degraded"
    parts.append(a)

    b = low.sort_values(["mean_prob", "rule_degraded_rate", "stop_share"], ascending=True).head(per_group)
    b["case_type"] = "LOW_SOC_low_warning_normal"
    parts.append(b)

    c = high.sort_values(["mean_prob", "rule_degraded_rate", "stop_share"], ascending=True).head(per_group)
    c["case_type"] = "HIGH_SOC_control_normal"
    parts.append(c)

    d = high.sort_values(["mean_prob", "rule_degraded_rate", "stop_share"], ascending=False).head(per_group)
    d["case_type"] = "HIGH_SOC_local_stop_or_false_warning"
    parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    cols = [
        "case_type", "session_role", "window_start_s", "window_end_s", "rows",
        "mean_soc", "min_soc", "max_soc", "mean_speed", "stop_share",
        "mean_prob", "max_prob", "warning_share", "high_risk_share", "rule_degraded_rate"
    ]
    return out[cols].sort_values(["case_type", "window_start_s"])


def plot_session_soc_speed(df: pd.DataFrame, out_dir: Path) -> None:
    p = out_dir / "clean_plots"
    ensure_dir(p)
    for role, g in df.groupby("session_role"):
        g = g.sort_values("t_s")
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(g["t_s"], g["battery_percent"], label="SOC (%)")
        ax1.set_xlabel("Time within session (s)")
        ax1.set_ylabel("SOC (%)")
        ax2 = ax1.twinx()
        ax2.plot(g["t_s"], g["speed"], alpha=0.5, label="Speed (m/s)")
        ax2.set_ylabel("Speed (m/s)")
        ax1.set_title(f"SOC and speed over time: {role}")
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")
        fig.tight_layout()
        fig.savefig(p / f"session_soc_speed_{role}.png", dpi=220)
        plt.close(fig)


def plot_case_windows(pred: pd.DataFrame, candidates: pd.DataFrame, out_dir: Path, max_cases: int = 8) -> None:
    p = out_dir / "clean_plots" / "case_windows"
    ensure_dir(p)
    chosen = candidates.groupby("case_type", group_keys=False).head(2).head(max_cases)
    rows = []
    for idx, row in chosen.reset_index(drop=True).iterrows():
        role = row["session_role"]
        a = float(row["window_start_s"])
        b = float(row["window_end_s"])
        sub = pred[(pred["session_role"].eq(role)) & (pred["t_s"].between(a - 10, b + 10))].copy()
        if sub.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(sub["t_s"], sub["y_prob"], label="Predicted probability")
        ax1.plot(sub["t_s"], sub["y_true"], alpha=0.7, label="Rule label")
        ax1.axhline(0.40, linestyle="--", linewidth=1, label="warning threshold")
        ax1.axhline(0.70, linestyle=":", linewidth=1, label="high-risk threshold")
        ax1.axvspan(a, b, alpha=0.15)
        ax1.set_xlabel("Time within session (s)")
        ax1.set_ylabel("Probability / label")
        ax2 = ax1.twinx()
        ax2.plot(sub["t_s"], sub["speed"], alpha=0.35, label="Speed (m/s)")
        ax2.set_ylabel("Speed (m/s)")
        ax1.set_title(f"{row['case_type']} | {role} | {int(a)}-{int(b)} s")
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")
        fig.tight_layout()
        fname = f"case_{idx+1}_{row['case_type']}_{role}_{int(a)}_{int(b)}.png"
        fig.savefig(p / fname, dpi=220)
        plt.close(fig)
        rows.append({**row.to_dict(), "plot_file": str(Path("clean_plots") / "case_windows" / fname)})
    pd.DataFrame(rows).to_csv(out_dir / "selected_case_study_windows.csv", index=False)


def make_advisory_replay_summary(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["warning"] = out["y_prob"] >= 0.40
    out["high_risk"] = out["y_prob"] >= 0.70
    out["charge_check"] = out["advisory"].eq("charge_check_or_safe_node")
    return out.groupby("session_role", as_index=False).agg(
        rows=("y_prob", "size"),
        true_degraded_rate=("y_true", "mean"),
        predicted_degraded_rate=("y_pred", "mean"),
        mean_prob=("y_prob", "mean"),
        warning_share=("warning", "mean"),
        high_risk_share=("high_risk", "mean"),
        charge_check_share=("charge_check", "mean"),
        mean_soc=("battery_percent", "mean"),
        min_soc=("battery_percent", "min"),
        mean_speed=("speed", "mean"),
        stop_share=("speed", lambda x: float((x < 0.03).mean())),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="low_soc_real_two_session_results")
    ap.add_argument("--window_s", type=int, default=30)
    ap.add_argument("--per_group", type=int, default=8)
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / "supplementary_outputs"
    ensure_dir(out_dir)

    processed = read_csv(results_dir / "processed_real_agv_1hz_features_labels_two_sessions.csv")
    pred = read_csv(results_dir / "advisory_predictions_low_train_high_control_h5.csv")

    win = make_window_table(pred, args.window_s)
    cand = pick_candidates(win, args.per_group)
    replay = make_advisory_replay_summary(pred)

    win.to_csv(out_dir / "all_manual_validation_windows.csv", index=False)
    cand.to_csv(out_dir / "manual_validation_candidate_windows.csv", index=False)
    replay.to_csv(out_dir / "advisory_replay_summary.csv", index=False)

    plot_session_soc_speed(processed, out_dir)
    plot_case_windows(pred, cand, out_dir)

    checklist = pd.DataFrame([
        {"item": "Inspect 20-30 candidate windows", "status": "to_fill_manually"},
        {"item": "Mark each window as normal / stop-go / poor-progress / false-warning", "status": "to_fill_manually"},
        {"item": "Use advisory_replay_summary.csv for advisory results", "status": "generated"},
    ])
    checklist.to_csv(out_dir / "manual_validation_checklist.csv", index=False)

    print("DONE")
    print(f"Output folder: {out_dir}")
    print("Main files:")
    print("  manual_validation_candidate_windows.csv")
    print("  advisory_replay_summary.csv")
    print("  clean_plots/session_soc_speed_*.png")
    print("  clean_plots/case_windows/*.png")


if __name__ == "__main__":
    main()
