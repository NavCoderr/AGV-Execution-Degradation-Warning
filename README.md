# Mission-Aware Short-Horizon Forecasting of AGV Execution Degradation

This repository contains the data-processing, modelling, evaluation, and audit scripts for short-horizon forecasting of execution-degradation onsets in graph-based automated guided vehicle (AGV) missions.

The monitored condition occurs when an AGV still has an active mission but is about to stop making effective physical progress. The formulation separates this condition from commanded waiting, safety holds, operator intervention, completed missions, unavailable telemetry, and already-degraded execution.

The repository evaluates whether directed-route context provides useful information beyond a common telemetry-and-provenance base and Euclidean target proximity when an entire physical recording session is held out.

The output is intended as a supervisory warning signal for possible TMS/MES decision support. It is not a mechanical-fault detector, certified safety controller, autonomous recovery policy, or battery-health estimator.

## Research question

The primary question is:

> Does directed-route context improve short-horizon forecasting of execution-degradation onsets beyond telemetry, observation provenance, and Euclidean target proximity when the model is evaluated on a completely unseen physical recording session?

The contribution is the monitoring formulation and leakage-controlled evaluation protocol rather than a new classifier architecture.

## Dataset

The evaluation uses five recording sessions collected from the same physical AGV and laboratory navigation layout.

The navigation graph contains:

- 35 nodes;
- 102 directed edges.

The five sessions contain 18,534 aligned one-second rows and 67 operational onsets under the fixed motion-and-mission definition.

| Session | Aligned rows | Observed | Short-held | Unavailable | Eligible 5-s rows | Onsets |
|---|---:|---:|---:|---:|---:|---:|
| S1 low-SOC stress | 8,027 | 4,240 | 3,444 | 343 | 2,148 | 25 |
| S2 high-SOC control | 1,679 | 790 | 801 | 88 | 387 | 0 |
| S3 medium-SOC | 2,288 | 2,288 | 0 | 0 | 1,159 | 10 |
| S4 safety-rich | 4,249 | 4,249 | 0 | 0 | 2,786 | 32 |
| S5 runtime-TMS | 2,291 | 1,190 | 966 | 135 | 724 | 0 |
| **Total** | **18,534** | **12,757** | **5,211** | **566** | **7,204** | **67** |

S2 and S5 contain no operational onset under the fixed rule and act as negative-control sessions.

Recorded routes cover 57.05% of the aligned rows, while 39.54% use a directed minimum-distance reconstruction. The remaining 3.41% do not have applicable graph context.

Aligned one-second rows are used to construct temporal histories. They are not treated as independent experimental repetitions. The primary unit of generalisation is a complete physical recording session.

## Input files

Place the input files inside the `data/` directory:

```text
data/
├── S1_LOW_SOC_STRESS.csv
├── S2_HIGH_SOC_CONTROL.csv
├── S3_MEDIUM_SOC_WHOLETESTING.csv
├── S4_SAFETY_RICH_NAVEEN12.csv
├── S5_REAL_TMS_JULY10.csv
├── Node_F3.csv
└── Edge_Distances3.csv
```

The scripts accept the recorded telemetry formats used in these sessions and harmonise them into a common representation.

## Past-only one-second harmonisation

Each recording is mapped onto an exact integer-second grid.

The harmonisation procedure:

1. validates timestamps and removes exact duplicate records;
2. aggregates observations falling within the same second;
3. propagates the latest past observation for at most two seconds;
4. marks longer communication gaps as unavailable;
5. records whether each row is observed, short-held, or unavailable;
6. prevents future samples from being used for interpolation;
7. separates discontinuous telemetry segments.

Unavailable telemetry is not interpreted as physical standstill. Short-held rows also do not count as genuine physical observations when constructing the degradation rule.

## Operational onset definition

The operational state rule uses a 10-second past-only history.

A candidate degraded state requires:

- an active mission during at least 60% of the history;
- an external explanation during no more than 20% of the history;
- either:
  - stop share of at least 0.60, using a speed threshold of 0.03 m/s; or
  - mean speed below 0.055 m/s;
- Cartesian displacement below 0.03 m;
- at least four genuine physical observations;
- at least six seconds of genuine observation span.

The candidate state must persist for three seconds before an onset is registered. A new onset can be registered only after ten seconds of recovery.

Forecast anchors are censored at:

- mission completion;
- an external hold;
- unavailable telemetry;
- the end of the mission leg;
- insufficient future follow-up.

Graph progress is not used in the target definition.

The target is an operational definition of ineffective mission execution. It is not an independently annotated mechanical-fault label.

## Route representation

If a valid controller-recorded route is available, it is retained.

When the route is unavailable, the pipeline reconstructs a minimum-distance directed path from the associated AGV position to the logged target node.

Route provenance is retained so that recorded and reconstructed route context are distinguishable.

Graph-derived variables include:

- graph-route remaining distance;
- route completion;
- recent graph progress;
- graph progress rate;
- current-edge progress and remaining distance;
- lateral graph-association error;
- edge dwell time;
- node degree and node-type indicators;
- route availability and provenance.

Euclidean variables include straight-line target distance and recent Euclidean progress.

## Feature representations

Four controlled representations are evaluated:

1. **Base context**
   - physical telemetry;
   - recent speed and stopping behaviour;
   - electrical and wheel measurements;
   - command-consistency measurements;
   - observation age and availability;
   - missingness indicators;
   - route-source provenance.

2. **Base + Euclidean**
   - base context;
   - Euclidean target distance;
   - recent Euclidean progress.

3. **Base + Euclidean + graph**
   - base and Euclidean context;
   - directed-route distance and completion;
   - recent graph progress;
   - current-edge and node context;
   - graph-association variables.

4. **Base + Euclidean + graph, no SOC**
   - the full representation;
   - all SOC-derived variables removed.

Active-mission and external-hold variables are used for eligibility and censoring. They are not predictive model inputs.

## Models

The tabular evaluation includes:

- Logistic Regression;
- ExtraTrees with 400 trees;
- Random Forest with 350 trees;
- Histogram Gradient Boosting with 250 iterations.

All learned models use random seed 42. Class-balanced learning is used where supported.

A separate temporal baseline evaluates a single-layer GRU using:

- a fixed 10-second sequence;
- 32 hidden units;
- dropout of 0.15;
- class-weighted binary cross-entropy;
- AdamW;
- learning rate of `1e-3`;
- weight decay of `1e-4`;
- batch size of 128;
- 40 training epochs;
- random seed 42.

The GRU is included as a temporal baseline. It is not presented as a novel neural architecture.

## Evaluation protocol

The primary protocol is complete-session leave-one-session-out evaluation.

For each fold:

1. one complete recording session is reserved for testing;
2. the other four sessions form the development data;
3. imputation, scaling, feature removal, and model fitting use development data only;
4. probability calibration excludes the test session;
5. the decision threshold is selected without using the test session.

When possible, a complete development session containing sufficient examples of both classes is reserved for Platt calibration. Otherwise, a class-valid chronological development tail is used.

The threshold is selected from 0.10 to 0.90 in increments of 0.02 using calibration-set macro-F1.

PR-AUC is the primary ranking metric because positive anchors are rare.

S2 and S5 do not contain positive onsets. PR-AUC is therefore undefined in those held-out folds and is averaged over the three positive-event sessions. Threshold-dependent metrics and Brier score are evaluated across all five sessions where applicable.

## Alert policy

A supervisory alert activates when at least two of the latest three calibrated probabilities exceed the development-selected threshold.

The policy also uses:

- a lower deactivation threshold;
- hysteresis;
- a five-second cooldown;
- state reset at session, telemetry-segment, mission-leg, or timestamp discontinuities.

An alert episode is matched when an onset occurs within the forecast horizon after the alert begins.

False-alert frequency is normalised by eligible monitored exposure rather than the complete wall-clock recording duration.

## Main strict pre-onset results

The strongest configuration at each horizon is:

| Horizon | Model and representation | PR-AUC | Macro-F1 | Positive F1 | Brier |
|---:|---|---:|---:|---:|---:|
| 5 s | Random Forest / Euclidean + graph, no SOC | 0.598 ± 0.267 | 0.674 ± 0.172 | 0.356 ± 0.352 | 0.016 ± 0.015 |
| 10 s | Random Forest / Euclidean + graph, no SOC | 0.675 ± 0.145 | 0.563 ± 0.102 | 0.152 ± 0.212 | 0.038 ± 0.042 |
| 20 s | Histogram Gradient Boosting / Euclidean | 0.597 ± 0.062 | 0.576 ± 0.102 | 0.211 ± 0.249 | 0.083 ± 0.081 |

PR-AUC is averaged across the three positive-event test sessions. The remaining metrics use all five held-out sessions.

### Five-second representation comparison

For Random Forest at the primary five-second horizon:

| Representation | Mean PR-AUC |
|---|---:|
| Base context | 0.488 ± 0.190 |
| Base + Euclidean | 0.587 ± 0.174 |
| Base + Euclidean + graph | 0.584 ± 0.259 |
| Base + Euclidean + graph, no SOC | 0.598 ± 0.267 |

The graph-related improvement is modest and varies across models and held-out sessions. The results do not support universal graph superiority.

## Alert replay results

At the selected five-second operating point, the Random Forest using Euclidean-plus-graph context without SOC:

- warns 45 of 67 operational onsets;
- achieves event-level recall of 0.672;
- produces 59 alert episodes;
- produces 15 false alert episodes;
- produces 7.50 false alerts per eligible forecasting hour;
- warns 34 events with at least two seconds of lead;
- warns 10 events with at least five seconds of lead;
- produces no alerts in the two negative-control sessions.

Session-level warned events are:

| Session | Warned / total |
|---|---:|
| S1 | 20 / 25 |
| S3 | 2 / 10 |
| S4 | 23 / 32 |

The alert remains suitable only for supervisory or shadow-mode evaluation.

## Temporal baseline results

The fixed-sequence GRU obtains mean PR-AUC values of:

| Horizon | GRU PR-AUC |
|---:|---:|
| 5 s | 0.404 ± 0.187 |
| 10 s | 0.530 ± 0.254 |
| 20 s | 0.352 ± 0.212 |

At five seconds, the GRU:

- warns 34 of 67 onsets;
- produces 27 false alert episodes;
- produces 13.49 false alerts per eligible forecasting hour.

The GRU does not improve the overall ranking and alert trade-off over the strongest tabular configuration in the available five-session dataset.

## Communication-gap stress test

A pre-specified ExtraTrees Euclidean-plus-graph model is tested after injecting communication gaps.

| Injected gap | Prediction coverage | PR-AUC |
|---:|---:|---:|
| 1 s | 1.000 | 0.519 |
| 2 s | 1.000 | 0.360 |
| 3 s | 0.997 | 0.467 |
| 5 s | 0.990 | 0.527 |
| 10 s | 0.974 | 0.402 |

The perturbation confirms that longer gaps reduce prediction coverage after the two-second hold allowance. It does not support a monotonic performance-degradation claim and does not reproduce every possible communication failure.

## Latency

Across the evaluated estimators and folds, the largest measured per-row p99 pipeline prediction time is approximately:

```text
0.477 ms
```

This measurement includes fitted in-pipeline preprocessing and probability prediction.

It does not include:

- telemetry acquisition;
- graph-state construction;
- communication delay;
- alert delivery;
- TMS/MES processing;
- control execution.

It is therefore not an end-to-end deployment latency guarantee.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── 01_build_scientific_dataset.py
├── 02_run_scientific_experiments.py
├── 03_run_pre_onset_experiments.py
├── 04_run_provenance_hazard.py
├── 05_run_temporal_gru_baseline.py
├── 06_prepare_acceptance_audits.py
├── 07_warning_figure.py
│
├── data/
│   ├── S1_LOW_SOC_STRESS.csv
│   ├── S2_HIGH_SOC_CONTROL.csv
│   ├── S3_MEDIUM_SOC_WHOLETESTING.csv
│   ├── S4_SAFETY_RICH_NAVEEN12.csv
│   ├── S5_REAL_TMS_JULY10.csv
│   ├── Node_F3.csv
│   └── Edge_Distances3.csv
│
├── figures/
│   └── final_successful_h5_warning.png
│
└── outputs/
    ├── harmonized_graph_mission_state.csv
    ├── event_onsets.csv
    ├── event_distribution.csv
    ├── label_definition.json
    ├── run_summary.json
    │
    ├── scientific_experiments/
    │   ├── 01_exact_1hz_verification.csv
    │   ├── 02_multi_horizon_loso_ablation.csv
    │   ├── 03_predictions.csv
    │   ├── 04_alert_policy_metrics.csv
    │   ├── 05_gap_robustness.csv
    │   ├── 06_latency.csv
    │   ├── 07_aggregate_results.csv
    │   ├── 08_representation_deltas.csv
    │   ├── 09_event_distribution.csv
    │   └── figures/
    │       └── final_h5_model_ablation.png
    │
    ├── pre_onset_experiments/
    │   ├── pre-onset LOSO result tables
    │   ├── aggregate summaries
    │   ├── held-out predictions
    │   └── best-model tables
    │
    ├── provenance_hazard/
    │   ├── hazard ablation results
    │   ├── cumulative-risk predictions
    │   └── coefficient tables
    │
    └── temporal_gru/
        ├── 01_temporal_gru_loso_results.csv
        ├── 02_temporal_gru_summary.csv
        ├── 03_temporal_gru_predictions.csv
        ├── 04_temporal_alert_tradeoffs.csv
        └── 05_temporal_vs_best_tabular.csv
```

The provenance-conditioned hazard experiment is an exploratory ablation and is not part of the primary claimed method.

The expert-audit script prepares review cases and forms. Running it does not itself constitute expert validation.

## Requirements

Python 3.10 or newer is recommended.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The requirements include:

- NumPy;
- pandas;
- scikit-learn;
- Matplotlib;
- NetworkX;
- OpenPyXL;
- xlrd;
- PyTorch.

## Run order

### 1. Build the harmonised scientific dataset

```bash
python 01_build_scientific_dataset.py
```

Primary output:

```text
outputs/harmonized_graph_mission_state.csv
```

### 2. Run the shared tabular experiments and engineering audits

```bash
python 02_run_scientific_experiments.py
```

This script also generates the model-ablation figure:

```text
outputs/scientific_experiments/figures/final_h5_model_ablation.png
```

### 3. Run the primary strict pre-onset evaluation

```bash
python 03_run_pre_onset_experiments.py
```

This is the primary script for the 5-, 10-, and 20-second strict pre-onset results.

### 4. Run the provenance-conditioned hazard ablation

```bash
python 04_run_provenance_hazard.py
```

This step is optional and is retained as an exploratory negative-result analysis.

### 5. Run the temporal GRU baseline

```bash
python 05_run_temporal_gru_baseline.py
```

PyTorch is required for this step.

### 6. Prepare expert-audit materials

```bash
python 06_prepare_acceptance_audits.py
```

This step prepares blinded cases and review forms. It does not generate an expert-validation result unless the forms are independently completed.

### 7. Generate the held-out warning figure

```bash
python 07_warning_figure.py
```

Input:

```text
outputs/pre_onset_experiments/09_pre_onset_predictions.csv
```

Output:

```text
figures/final_successful_h5_warning.png
```

## Reproducibility notes

- Random seed 42 is used throughout the learned experiments.
- Complete recording sessions are held out during testing.
- Calibration and threshold selection exclude the test session.
- Sequences do not cross session, telemetry-segment, mission-leg, or timestamp discontinuities.
- All-missing features are removed using fitting data only.
- Numeric imputation and scaling are fitted on development data only.
- Negative-only sessions remain included in alert, Brier-score, and threshold-dependent evaluation.
- Graph progress is excluded from the target definition.
- Result rows are not interpreted as independent physical experiments.
