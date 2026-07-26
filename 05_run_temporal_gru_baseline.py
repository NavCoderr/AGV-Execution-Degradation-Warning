from __future__ import annotations

"""

Dependency:
    python3 -m pip install torch
"""

import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for the temporal GRU baseline.\n"
        "Install it with:\n"
        "    python3 -m pip install torch\n"
        "Then run this script again."
    ) from exc


RANDOM_STATE = 42
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def load_experiment_module(path: Path):
    spec = importlib.util.spec_from_file_location("agv_experiments", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment utilities: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TemporalGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.15)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


def add_fixed_provenance_channels(data: pd.DataFrame) -> pd.DataFrame:
    """Encode route provenance without learning categories from test data."""
    out = data.copy()
    source = out["route_source"].fillna("unavailable").astype(str)
    out["route_recorded"] = source.eq("recorded").astype(float)
    out["route_reconstructed"] = source.eq(
        "reconstructed_shortest"
    ).astype(float)
    out["route_context_unavailable"] = (
        1.0 - out["route_recorded"] - out["route_reconstructed"]
    ).clip(0.0, 1.0)
    return out


def build_sequence_index(data: pd.DataFrame) -> dict[int, np.ndarray]:
    """Map each eligible row id to its exact, within-leg history row ids."""
    sequence_index: dict[int, np.ndarray] = {}
    group_columns = [
        "session",
        "temporal_segment_id",
        "mission_leg_id",
    ]
    for _, group in data.groupby(
        group_columns,
        sort=False,
        dropna=True,
    ):
        group = group.sort_values("timestamp")
        row_ids = group["_row_id"].to_numpy(dtype=int)
        timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        for end in range(SEQUENCE_LENGTH - 1, len(group)):
            start = end - SEQUENCE_LENGTH + 1
            window_times = timestamps[start : end + 1]
            differences = np.diff(window_times).astype("timedelta64[s]").astype(int)
            if differences.size and not np.all(differences == 1):
                continue
            sequence_index[int(row_ids[end])] = row_ids[start : end + 1].copy()
    return sequence_index


def sequences_for_anchors(
    data_by_id: pd.DataFrame,
    sequence_index: dict[int, np.ndarray],
    anchors: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return sequences and row ids in deterministic anchor order."""
    arrays: list[np.ndarray] = []
    retained_ids: list[int] = []
    for row_id in anchors["_row_id"].astype(int).tolist():
        history_ids = sequence_index.get(row_id)
        if history_ids is None:
            continue
        window = (
            data_by_id.loc[history_ids, features]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        if window.shape != (SEQUENCE_LENGTH, len(features)):
            continue
        arrays.append(window)
        retained_ids.append(row_id)
    if not arrays:
        return (
            np.empty((0, SEQUENCE_LENGTH, len(features)), dtype=np.float32),
            np.empty(0, dtype=int),
        )
    return np.stack(arrays), np.asarray(retained_ids, dtype=int)


def remove_all_missing_training_channels(
    x_fit: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Remove channels using fitting data only."""
    available = np.isfinite(x_fit).any(axis=(0, 1))
    kept = [feature for feature, keep in zip(features, available) if keep]
    removed = [feature for feature, keep in zip(features, available) if not keep]
    return available, kept, removed


def fit_sequence_scaler(
    x_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit channel-wise median imputation and standardization on fit only."""
    flat = x_fit.reshape(-1, x_fit.shape[-1]).astype(float)
    median = np.nanmedian(flat, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    imputed = np.where(np.isfinite(flat), flat, median)
    mean = imputed.mean(axis=0)
    scale = imputed.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return median.astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def transform_sequences(
    x: np.ndarray,
    median: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    imputed = np.where(np.isfinite(x), x, median)
    return ((imputed - mean) / scale).astype(np.float32)


def train_gru(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    device: torch.device,
) -> tuple[TemporalGRU, float]:
    set_seed(RANDOM_STATE)
    model = TemporalGRU(input_size=x_fit.shape[-1]).to(device)

    positives = max(int(y_fit.sum()), 1)
    negatives = max(int(len(y_fit) - y_fit.sum()), 1)
    positive_weight = min(float(negatives / positives), 30.0)
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    dataset = TensorDataset(
        torch.from_numpy(x_fit),
        torch.from_numpy(y_fit.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model.train()
    final_loss = np.nan
    for _ in range(EPOCHS):
        epoch_loss = 0.0
        epoch_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * len(batch_x)
            epoch_rows += len(batch_x)
        final_loss = epoch_loss / max(epoch_rows, 1)
    return model, float(final_loss)


def predict_probability(
    model: TemporalGRU,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probabilities: list[np.ndarray] = []
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(
        dataset,
        batch_size=min(512, max(len(dataset), 1)),
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(float)


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "f1_positive",
        "roc_auc",
        "pr_auc",
        "brier",
        "positive_rate",
    ]
    rows: list[dict] = []
    for horizon, group in results.groupby("horizon_s", sort=True):
        row = {
            "horizon_s": int(horizon),
            "model": "TemporalGRU10s",
            "feature_set": "full_graph_no_soc",
            "folds": int(group["test_session"].nunique()),
            "test_rows_total": int(group["n"].sum()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_evaluable_folds"] = int(values.notna().sum())
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_alert_tradeoffs(
    predictions: pd.DataFrame,
    exp,
) -> pd.DataFrame:
    """Report sensitive, learned, and conservative alert operating points."""
    profiles = {
        "sensitive_0.8x": 0.8,
        "learned_1.0x": 1.0,
        "conservative_1.2x": 1.2,
    }
    rows: list[dict] = []
    for (horizon, session), fold in predictions.groupby(
        ["horizon_s", "session"],
        sort=True,
    ):
        label = f"label_h{int(horizon)}"
        for profile, multiplier in profiles.items():
            work = fold.copy()
            learned_threshold = float(work["threshold"].iloc[0])
            threshold = float(np.clip(learned_threshold * multiplier, 0.02, 0.95))
            work["alert"] = exp.segmented_alert_policy(work, on=threshold)
            metrics = exp.alert_metrics(work, label, int(horizon))
            metrics.update(
                {
                    "horizon_s": int(horizon),
                    "test_session": session,
                    "operating_point": profile,
                    "threshold": threshold,
                    "evaluated_hours": float(len(work) / 3600.0),
                }
            )
            rows.append(metrics)

    per_fold = pd.DataFrame(rows)
    aggregate_rows: list[dict] = []
    for (horizon, profile), group in per_fold.groupby(
        ["horizon_s", "operating_point"],
        sort=True,
    ):
        events = int(group["events"].sum())
        warned = int(group["warned_events"].sum())
        false_alerts = int(group["false_alert_episodes"].sum())
        hours = float(group["evaluated_hours"].sum())
        aggregate_rows.append(
            {
                "horizon_s": int(horizon),
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
    return pd.DataFrame(aggregate_rows)


def compare_with_tabular(
    temporal_summary: pd.DataFrame,
    tabular_file: Path,
) -> pd.DataFrame:
    if not tabular_file.exists():
        return pd.DataFrame()
    tabular = pd.read_csv(tabular_file)
    columns = [
        "horizon_s",
        "model",
        "feature_set",
        "folds",
        "pr_auc_mean",
        "pr_auc_sd",
        "macro_f1_mean",
        "macro_f1_sd",
        "f1_positive_mean",
        "f1_positive_sd",
        "brier_mean",
        "brier_sd",
    ]
    columns = [column for column in columns if column in tabular.columns]
    tabular = tabular[columns].copy()
    tabular["comparison_role"] = "best_tabular_from_existing_experiment"

    temporal_columns = [
        column for column in columns if column in temporal_summary.columns
    ]
    temporal = temporal_summary[temporal_columns].copy()
    temporal["comparison_role"] = "pre_specified_temporal_baseline"
    return pd.concat([tabular, temporal], ignore_index=True, sort=False)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir / "outputs" / "harmonized_graph_mission_state.csv"
    base_script = script_dir / "02_run_scientific_experiments.py"
    tabular_best_file = (
        script_dir
        / "outputs"
        / "pre_onset_experiments"
        / "10_pre_onset_best_models.csv"
    )
    out_dir = script_dir / "outputs" / "temporal_gru"

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}\n"
            "Run 01_build_scientific_dataset.py first."
        )
    if not base_script.exists():
        raise FileNotFoundError(
            f"Base experiment script not found: {base_script}"
        )

    exp = load_experiment_module(base_script)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_STATE)

    data = pd.read_csv(data_file, parse_dates=["timestamp"], low_memory=False)
    required = {
        "session",
        "timestamp",
        "temporal_segment_id",
        "mission_leg_id",
        "active_mission",
        "external_hold",
        "telemetry_fresh",
        "at_risk",
        "route_source",
        "time_to_onset_s",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing required columns: {missing}")

    data = data.sort_values(["session", "timestamp"]).reset_index(drop=True)
    data["_row_id"] = np.arange(len(data), dtype=int)
    data = add_fixed_provenance_channels(data)
    data_by_id = data.set_index("_row_id", drop=False)
    sequence_index = build_sequence_index(data)

    full_graph_no_soc = [
        feature
        for feature in (exp.BASE + exp.GRAPH + exp.EUCLID)
        if feature not in exp.SOC_FEATURES and feature in data.columns
    ]
    provenance_channels = [
        "route_recorded",
        "route_reconstructed",
        "route_context_unavailable",
    ]
    requested_features = list(
        dict.fromkeys(full_graph_no_soc + provenance_channels)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon in (5, 10, 20):
        label = f"label_h{horizon}"
        if label not in data.columns:
            continue
        eligible = data[
            data["at_risk"].eq(1)
            & data["active_mission"].eq(1)
            & data["external_hold"].eq(0)
            & data["telemetry_fresh"].eq(1)
            & data[label].notna()
            & data["_row_id"].isin(sequence_index)
        ].copy()

        for test_session in sorted(eligible["session"].dropna().unique()):
            test_anchors = eligible[eligible["session"].eq(test_session)].copy()
            development = eligible[eligible["session"].ne(test_session)].copy()
            if len(test_anchors) < 20 or development[label].nunique() < 2:
                continue

            try:
                fit_anchors, calibration_anchors, calibration_source = (
                    exp.split_fit_calibration(development, label)
                )
            except RuntimeError:
                continue

            x_fit, fit_ids = sequences_for_anchors(
                data_by_id,
                sequence_index,
                fit_anchors,
                requested_features,
            )
            x_cal, cal_ids = sequences_for_anchors(
                data_by_id,
                sequence_index,
                calibration_anchors,
                requested_features,
            )
            x_test, test_ids = sequences_for_anchors(
                data_by_id,
                sequence_index,
                test_anchors,
                requested_features,
            )
            if min(len(x_fit), len(x_cal), len(x_test)) == 0:
                continue

            fit_lookup = fit_anchors.set_index("_row_id")
            calibration_lookup = calibration_anchors.set_index("_row_id")
            test_lookup = test_anchors.set_index("_row_id")
            y_fit = fit_lookup.loc[fit_ids, label].astype(int).to_numpy()
            y_cal = calibration_lookup.loc[cal_ids, label].astype(int).to_numpy()
            y_test = test_lookup.loc[test_ids, label].astype(int).to_numpy()
            if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
                continue

            channel_mask, features, removed_features = (
                remove_all_missing_training_channels(x_fit, requested_features)
            )
            x_fit = x_fit[:, :, channel_mask]
            x_cal = x_cal[:, :, channel_mask]
            x_test = x_test[:, :, channel_mask]

            median, mean, scale = fit_sequence_scaler(x_fit)
            x_fit = transform_sequences(x_fit, median, mean, scale)
            x_cal = transform_sequences(x_cal, median, mean, scale)
            x_test = transform_sequences(x_test, median, mean, scale)

            start = time.perf_counter()
            model, final_training_loss = train_gru(x_fit, y_fit, device)
            fit_seconds = time.perf_counter() - start

            p_cal_raw = predict_probability(model, x_cal, device)
            calibrator, calibration_method = exp.fit_platt_calibrator(
                y_cal,
                p_cal_raw,
            )
            p_cal = exp.apply_probability_calibrator(calibrator, p_cal_raw)
            threshold = exp.select_threshold(y_cal, p_cal)

            start = time.perf_counter()
            p_test_raw = predict_probability(model, x_test, device)
            p_test = exp.apply_probability_calibrator(calibrator, p_test_raw)
            infer_ms_row = (
                (time.perf_counter() - start) * 1000.0 / max(len(x_test), 1)
            )

            metrics = exp.score_binary(y_test, p_test, threshold)
            metrics.update(
                {
                    "task": "strict_pre_onset_temporal_sequence",
                    "horizon_s": horizon,
                    "variant": "all_sessions",
                    "feature_set": "full_graph_no_soc",
                    "test_session": test_session,
                    "model": "TemporalGRU10s",
                    "sequence_length_s": SEQUENCE_LENGTH,
                    "calibration_source": calibration_source,
                    "calibration_method": calibration_method,
                    "fit_rows": int(len(x_fit)),
                    "calibration_rows": int(len(x_cal)),
                    "test_rows": int(len(x_test)),
                    "fit_s": float(fit_seconds),
                    "infer_ms_row": float(infer_ms_row),
                    "final_training_loss": final_training_loss,
                    "n_used_features": int(len(features)),
                    "removed_all_missing_features": "|".join(removed_features),
                    "device": str(device),
                }
            )
            result_rows.append(metrics)

            prediction_columns = [
                "session",
                "timestamp",
                "temporal_segment_id",
                "mission_leg_id",
                "time_to_onset_s",
                label,
            ]
            prediction = test_lookup.loc[test_ids, prediction_columns].copy()
            prediction["horizon_s"] = horizon
            prediction["variant"] = "all_sessions"
            prediction["feature_set"] = "full_graph_no_soc"
            prediction["model"] = "TemporalGRU10s"
            prediction["probability"] = p_test
            prediction["threshold"] = threshold
            prediction["prediction"] = (p_test >= threshold).astype(int)
            prediction_frames.append(prediction.reset_index(drop=True))

    results = pd.DataFrame(result_rows)
    if results.empty:
        raise RuntimeError(
            "No temporal LOSO folds were produced. Check event counts, "
            "sequence eligibility, and calibration partitions."
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = aggregate_results(results)
    tradeoffs = make_alert_tradeoffs(predictions, exp)
    comparison = compare_with_tabular(summary, tabular_best_file)

    results.to_csv(out_dir / "01_temporal_gru_loso_results.csv", index=False)
    summary.to_csv(out_dir / "02_temporal_gru_summary.csv", index=False)
    predictions.to_csv(out_dir / "03_temporal_gru_predictions.csv", index=False)
    tradeoffs.to_csv(out_dir / "04_temporal_alert_tradeoffs.csv", index=False)
    if not comparison.empty:
        comparison.to_csv(
            out_dir / "05_temporal_vs_best_tabular.csv",
            index=False,
        )

    manifest = {
        "model": "single-layer GRU with fixed 10-second sequence",
        "purpose": "temporal baseline; no novel architecture claim",
        "primary_protocol": "complete-session leave-one-session-out",
        "sequence_boundary_rule": (
            "no crossing of session, telemetry segment, mission leg, "
            "or non-1-second timestamp interval"
        ),
        "preprocessing": (
            "channel removal, median imputation, and standardization "
            "fitted on the fitting partition only"
        ),
        "calibration": "development-only Platt calibration",
        "threshold_selection": "development-only macro-F1",
        "feature_set": "full graph without SOC plus fixed provenance channels",
        "sequence_length_s": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "hidden_size": HIDDEN_SIZE,
        "random_state": RANDOM_STATE,
        "device": str(device),
        "caution": (
            "This baseline addresses temporal-model completeness but does "
            "not solve limited external validity or automatic-label validity."
        ),
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Temporal GRU experiment completed.")
    print(summary.to_string(index=False))
    print(f"\nResults written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
