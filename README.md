# Mission-Aware Short-Horizon Forecasting of AGV Execution Degradation

This repository contains the data-processing, modelling, evaluation, and robustness scripts associated with short-horizon forecasting of execution-degradation onsets in graph-based automated guided vehicle (AGV) missions.

The study addresses a post-dispatch monitoring problem: given an active AGV mission, current physical observations, and available route context, can an upcoming loss of effective physical progress be forecast before the confirmed onset?

The main research question is whether **directed-route context provides predictive information beyond physical telemetry, observation provenance, and Euclidean target proximity when an entire physical recording session is unseen**.

The proposed output is a supervisory warning signal intended for possible TMS/MES decision support. It is not a mechanical-fault detector, certified safety controller, or autonomous recovery policy.

---

## Dataset

The evaluation uses five physical recording sessions collected from the same AGV and laboratory navigation layout.

The navigation graph contains:

- 35 nodes
- 102 directed edges

After past-only one-second harmonisation, the dataset contains:

- **18,534 aligned rows**
- **67 operational onsets**
- **7,204 eligible 5-s forecasting rows**
- **6,151 eligible 10-s forecasting rows**
- **4,644 eligible 20-s forecasting rows**

| Session | Aligned rows | Eligible 5-s rows | Onsets |
| --- | ---: | ---: | ---: |
| S1 low-SOC stress | 8,027 | 2,148 | 25 |
| S2 high-SOC control | 1,679 | 387 | 0 |
| S3 medium-SOC | 2,288 | 1,159 | 10 |
| S4 safety-rich | 4,249 | 2,786 | 32 |
| S5 runtime-TMS | 2,291 | 724 | 0 |
| **Total** | **18,534** | **7,204** | **67** |

S1, S3, and S4 contain operational onsets. S2 and S5 contain no operational onset under the fixed rule and are retained as negative-control sessions.

Recorded routes cover 57.05% of aligned rows, while 39.54% use a minimum-distance directed reconstruction. The remaining 3.41% do not have applicable graph context.

For the operational onset events:

- S1 uses recorded route context
- S3 and S4 use reconstructed directed-route context because controller-recorded routes are unavailable for those events

Route-source provenance is retained explicitly.

---

## Input files

Place the source files in `data/`:

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

---

## Past-only harmonisation

Each recording session is mapped to an integer-second analysis grid.

The preprocessing pipeline:

1. validates timestamps and removes exact duplicate records
2. aggregates observations within the same second
3. propagates the latest past observation for at most 2 s
4. marks longer gaps as unavailable
5. records observation provenance
6. does not use future samples for interpolation
7. separates discontinuous telemetry segments

Unavailable telemetry is not interpreted as physical standstill.

The one-second representation refers to the harmonised analysis grid and does not imply that every original telemetry source was natively sampled at exactly 1 Hz.

---

## Operational onset definition

The operational state rule uses a 10-s past-only history.

A candidate degraded state requires:

- an active mission during at least 60% of the history
- no dominant external explanation
- stop share at least 0.60 using a speed threshold of 0.03 m/s, or mean speed below 0.055 m/s
- Cartesian displacement below 0.03 m
- at least four genuine physical observations
- at least 6 s of genuine observation span

The candidate state must persist for 3 s before an onset is registered.

A new onset can be registered only after 10 s of recovery.

Onset therefore denotes the **confirmation time of a sustained degradation episode**. The final pre-onset samples may contain unconfirmed degradation cues, but no confirmed degradation episode is active at the forecasting anchor.

For forecasting horizon `h`, the target is positive when the first confirmed onset occurs within:

```text
(t, t + h]
```

for:

```text
h ∈ {5, 10, 20} s
```

Forecast anchors are censored at mission completion, external holds, unavailable telemetry, mission-leg boundaries, and insufficient future follow-up.

Graph progress is not used in the target definition.

---

## Feature representations

Five controlled feature representations are evaluated.

### 1. Base context

Includes:

- physical telemetry
- speed and stopping history
- electrical and wheel measurements
- command-consistency variables
- position confidence
- observation age
- telemetry availability
- missingness indicators
- route-source provenance

### 2. Base + Euclidean

Adds:

- straight-line target distance
- recent Euclidean progress

### 3. Base + Euclidean, no SOC

Uses the Euclidean representation after removing all SOC-derived variables.

### 4. Base + Euclidean + graph

Adds:

- directed-route remaining distance
- route completion
- recent graph progress
- current-edge context
- node context
- graph-association variables

### 5. Base + Euclidean + graph, no SOC

Uses the graph representation after removing all SOC-derived variables.

Active-mission and external-hold variables are used for eligibility and censoring and are not predictive model inputs.

---

## Models

The tabular evaluation includes:

- Logistic Regression
- ExtraTrees, 400 trees
- Random Forest, 350 trees
- Histogram Gradient Boosting, 250 iterations

All learned models use random seed 42.

A separate temporal baseline uses a single-layer GRU with:

- 10-s input sequence
- 32 hidden units
- dropout 0.15
- class-weighted binary cross-entropy
- AdamW
- learning rate `1e-3`
- weight decay `1e-4`
- batch size 128
- 40 epochs
- random seed 42

The GRU is included as a temporal baseline and is not presented as a novel neural architecture.

---

## Evaluation protocol

The primary protocol is **complete-session leave-one-session-out (LOSO)** evaluation.

For every fold:

1. one complete physical recording session is reserved for testing
2. all remaining sessions form the development data
3. imputation, scaling, feature removal, and model fitting use development data only
4. probability calibration excludes the held-out session
5. the probability threshold is selected without using the held-out session

PR-AUC is the primary ranking metric because positive forecasting anchors are rare.

S2 and S5 contain no positive events, so PR-AUC is undefined for these two held-out folds.

PR-AUC summaries are therefore averaged across the three positive-event held-out sessions.

---

## Primary 5-s result

The primary reported analysis keeps the model fixed as Random Forest and compares matched feature representations.

| Representation | Mean PR-AUC |
| --- | ---: |
| Base context | 0.488 ± 0.190 |
| Base + Euclidean | 0.587 ± 0.174 |
| **Base + Euclidean, no SOC** | **0.570 ± 0.214** |
| Base + Euclidean + graph | 0.584 ± 0.259 |
| **Base + Euclidean + graph, no SOC** | **0.598 ± 0.267** |

The primary matched no-SOC comparison is therefore:

```text
Euclidean, no SOC:
0.570 ± 0.214

Euclidean + graph, no SOC:
0.598 ± 0.267

Difference:
+0.028
```

At the development-selected operating points, the same matched comparison changes:

- macro-F1: 0.604 → 0.674
- positive-class F1: 0.221 → 0.356
- recall: 0.190 → 0.325
- Brier score: 0.0162 → 0.0157

The improvement is interpreted as complementary route-context information rather than universal graph superiority.

---

## Matched comparison across horizons

To examine whether the representation effect persists with forecast horizon, both the Random Forest model and no-SOC condition are kept fixed.

| Horizon | Euclidean, no SOC | Euclidean + graph, no SOC | Difference |
| ---: | ---: | ---: | ---: |
| 5 s | 0.570 ± 0.214 | 0.598 ± 0.267 | +0.028 |
| 10 s | 0.636 ± 0.114 | 0.675 ± 0.145 | +0.039 |
| 20 s | 0.527 ± 0.172 | 0.494 ± 0.247 | -0.034 |

At 10 s, fold-level PR-AUC changes from:

```text
S1: 0.709 → 0.735
S3: 0.504 → 0.509
S4: 0.694 → 0.780
```

Thus, the graph representation improves PR-AUC in all three positive-event held-out sessions at 10 s.

The graph advantage does not persist at 20 s.

---

## Target-feature sensitivity

Run:

```bash
python 08_run_target_feature_sensitivity.py
```

This experiment repeats the matched no-SOC Random Forest comparison after removing the direct speed and stop-history predictors:

```text
speed_mps
speed_mean_5s
speed_mean_10s
speed_mean_30s
speed_std_10s
stop_share_5s
stop_share_10s
stop_share_30s
```

Results:

| Horizon | Euclidean, no SOC | Euclidean + graph, no SOC |
| ---: | ---: | ---: |
| 5 s | 0.538 ± 0.161 | 0.593 ± 0.291 |
| 10 s | 0.634 ± 0.128 | 0.661 ± 0.173 |

This provides additional evidence that the observed short-horizon route-context ranking advantage is not explained solely by the direct speed and stop-history predictors.

Outputs:

```text
outputs/target_feature_sensitivity/
├── 01_loso_results.csv
├── 02_summary.csv
└── 03_motion_history_sensitivity.png
```

---

## Candidate-free-anchor sensitivity

Run Step 3 first, then:

```bash
python 09_run_pre_candidate_sensitivity.py
```

This post-hoc robustness analysis uses already-generated held-out Random Forest predictions and restricts evaluation to anchors where the candidate-degradation rule is not active.

Results:

| Horizon | Euclidean, no SOC | Euclidean + graph, no SOC |
| ---: | ---: | ---: |
| 5 s | 0.548 | 0.595 |
| 10 s | 0.747 | 0.798 |
| 20 s | 0.492 | 0.457 |

Within this stricter subset, the graph representation retains higher mean PR-AUC at 5 and 10 s.

The advantage does not persist at 20 s.

This is a post-hoc robustness analysis and is not treated as a separate confirmatory task.

Outputs:

```text
outputs/pre_candidate_sensitivity/
├── 01_fold_results.csv
├── 02_summary.csv
└── 03_pre_candidate_sensitivity.png
```

---

## Exploratory highest-observed configurations

For completeness, the highest observed mean PR-AUC among all evaluated model-representation combinations is:

| Horizon | Model / representation | Mean PR-AUC |
| ---: | --- | ---: |
| 5 s | Random Forest / Euclidean + graph, no SOC | 0.598 ± 0.267 |
| 10 s | Histogram Gradient Boosting / Euclidean, no SOC | 0.686 ± 0.183 |
| 20 s | Histogram Gradient Boosting / Euclidean | 0.597 ± 0.062 |

These configurations were identified from the held-out LOSO results and are therefore reported only as exploratory observations.

They are not used as independently selected confirmatory test configurations.

---

## Alert replay

At the primary 5-s operating point, the Random Forest using Euclidean-plus-graph context without SOC:

- warns 45 of 67 operational onsets
- achieves event-level recall of 0.672
- produces 59 alert episodes
- produces 15 false alert episodes
- produces 7.50 false alert episodes per eligible forecasting hour
- warns 34 events with at least 2 s lead
- warns 10 events with at least 5 s lead
- produces no alert episodes in S2 or S5

Session-level warned events:

| Session | Warned / total |
| --- | ---: |
| S1 | 20 / 25 |
| S3 | 2 / 10 |
| S4 | 23 / 32 |

The alert is intended for supervisory evaluation rather than autonomous control.

### Operating-point sensitivity

The alert policy is also replayed after scaling each development-selected threshold without selecting a new threshold from the held-out session.

| Operating point | Warned | False-alert frequency |
| --- | ---: | ---: |
| Conservative, 1.2x | 39 / 67 | 3.50 episodes/h |
| Learned, 1.0x | 45 / 67 | 7.50 episodes/h |
| Sensitive, 0.8x | 46 / 67 | 9.49 episodes/h |

These values illustrate the supervisory recall-alert-burden trade-off.

---

## Temporal GRU baseline

The fixed-sequence GRU obtains:

| Horizon | Mean PR-AUC |
| ---: | ---: |
| 5 s | 0.404 ± 0.187 |
| 10 s | 0.530 ± 0.254 |
| 20 s | 0.352 ± 0.212 |

At the primary 5-s horizon, the GRU:

- warns 34 of 67 operational onsets
- produces 27 false alert episodes
- produces 13.49 false alerts per eligible forecasting hour

Under the available five-session dataset, the pre-specified GRU provides a less favourable primary-horizon ranking and alert trade-off than the reported Random Forest graph representation.

This does not imply that recurrent or other temporal models are generally inferior.

---

## Communication-gap robustness

`02_run_scientific_experiments.py` also evaluates an ExtraTrees Euclidean-plus-graph model after injecting communication gaps of:

```text
1, 2, 3, 5, and 10 s
```

Prediction coverage decreases as gaps become longer after the two-second hold allowance.

The complete numerical results are stored in:

```text
outputs/scientific_experiments/05_gap_robustness.csv
```

This stress test does not reproduce every possible delay, packet-loss, reordering, or network-failure condition.

---

## Latency

The largest measured per-row p99 pipeline prediction time across the evaluated models and folds is approximately:

```text
0.535 ms
```

This includes fitted in-pipeline preprocessing, transformation, and probability prediction.

It does not include:

- telemetry acquisition
- graph-state construction
- communication
- alert delivery
- TMS/MES processing
- control execution

It is therefore not an end-to-end latency guarantee.

---

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
├── 06_prepare_expert_audit.py
├── 07_warning_figure.py
├── 08_run_target_feature_sensitivity.py
├── 09_run_pre_candidate_sensitivity.py
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
    ├── scientific_experiments/
    ├── pre_onset_experiments/
    ├── target_feature_sensitivity/
    ├── pre_candidate_sensitivity/
    ├── provenance_hazard/
    ├── temporal_gru/
    └── acceptance_audits/
```

---

## Generated and large output files

Some experiment outputs, especially prediction-level CSV files, can be large.

GitHub's browser-based uploader does not accept files larger than its web-upload limit. Therefore, some large generated files may not be committed in the repository snapshot.

This does **not** affect the required input data or experiment definitions.

Missing generated outputs can be recreated by running the corresponding experiment script.

The committed summary, aggregate, and figure outputs provide compact records of the reported results.

---

## Requirements

Python 3.10 or newer is recommended.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

PyTorch is additionally required for the GRU baseline.

---

## Run order

### Step 1 — Build the harmonised dataset

```bash
python 01_build_scientific_dataset.py
```

Primary output:

```text
outputs/harmonized_graph_mission_state.csv
```

### Step 2 — Run shared tabular experiments

```bash
python 02_run_scientific_experiments.py
```

This generates the LOSO tabular results, representation comparisons, communication-gap analysis, alert-policy results, and latency measurements.

### Step 3 — Run strict pre-onset experiments

```bash
python 03_run_pre_onset_experiments.py
```

Important outputs include:

```text
outputs/pre_onset_experiments/07_pre_onset_loso_results.csv
outputs/pre_onset_experiments/08_pre_onset_summary.csv
outputs/pre_onset_experiments/09_pre_onset_predictions.csv
outputs/pre_onset_experiments/10_pre_onset_best_models.csv
```

### Step 4 — Run exploratory provenance-hazard analysis

```bash
python 04_run_provenance_hazard.py
```

This is exploratory and is not part of the primary claimed method.

### Step 5 — Run temporal GRU baseline

```bash
python 05_run_temporal_gru_baseline.py
```

### Step 6 — Prepare expert-audit material

```bash
python 06_prepare_expert_audit.py
```

This prepares blinded review cases and forms. Running the script does not itself constitute independent expert validation.

### Step 7 — Generate held-out warning figure

```bash
python 07_warning_figure.py
```

### Step 8 — Run target-feature sensitivity

```bash
python 08_run_target_feature_sensitivity.py
```

### Step 9 — Run candidate-free-anchor sensitivity

```bash
python 09_run_pre_candidate_sensitivity.py
```

Step 3 must be completed first because Step 9 uses the held-out prediction file produced by the strict pre-onset experiment.
