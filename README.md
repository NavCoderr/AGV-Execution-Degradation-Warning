# Mission-Aware Short-Horizon Forecasting of AGV Execution Degradation

This repository contains the data-processing, modelling, evaluation, and audit scripts for short-horizon forecasting of execution-degradation onsets in graph-based automated guided vehicle (AGV) missions.

The monitored condition occurs when an AGV retains an active mission but is about to lose effective physical progress. The formulation distinguishes this condition from commanded waiting, safety holds, operator intervention, mission completion, unavailable telemetry, and already-degraded execution.

The repository evaluates whether directed-route context provides predictive information beyond a common telemetry-and-provenance representation and Euclidean target proximity when an entire physical recording session is held out.

The output is intended as a supervisory warning signal for possible TMS/MES decision support. It is not a mechanical-fault detector, certified safety controller, or autonomous recovery policy.

## Research question

The primary question is:

> Does directed-route context improve short-horizon forecasting of execution-degradation onsets beyond telemetry, observation provenance, and Euclidean target proximity when the model is evaluated on a completely unseen physical recording session?

The contribution is the post-dispatch monitoring formulation, controlled route-context comparison, and complete-session evaluation protocol rather than a new classifier architecture.

## Dataset

The evaluation uses five physical recording sessions collected from the same AGV and laboratory navigation layout.

The navigation graph contains:

* 35 nodes;
* 102 directed edges.

The five sessions contain 18,534 aligned one-second rows and 67 operational onsets under the fixed motion-and-mission definition.

| Session             | Aligned rows |   Observed | Short-held | Unavailable | Eligible 5-s rows | Onsets |
| ------------------- | -----------: | ---------: | ---------: | ----------: | ----------------: | -----: |
| S1 low-SOC stress   |        8,027 |      4,240 |      3,444 |         343 |             2,148 |     25 |
| S2 high-SOC control |        1,679 |        790 |        801 |          88 |               387 |      0 |
| S3 medium-SOC       |        2,288 |      2,288 |          0 |           0 |             1,159 |     10 |
| S4 safety-rich      |        4,249 |      4,249 |          0 |           0 |             2,786 |     32 |
| S5 runtime-TMS      |        2,291 |      1,190 |        966 |         135 |               724 |      0 |
| **Total**           |   **18,534** | **12,757** |  **5,211** |     **566** |         **7,204** | **67** |

S1, S3, and S4 contain operational onsets. S2 and S5 contain no operational onset under the fixed rule and are retained as negative-control sessions.

Recorded routes cover 57.05% of the aligned rows, while 39.54% use a directed minimum-distance reconstruction. The remaining 3.41% do not have applicable graph context.

For the operational onset events, S1 uses recorded route context. Recorded controller routes are unavailable for the operational onset events in S3 and S4; their directed-route context is therefore based on minimum-distance directed reconstruction.

Route provenance is retained explicitly so that recorded and reconstructed route context remain distinguishishable.

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

## Past-only one-second harmonisation

Each physical recording session is mapped onto an integer-second analysis grid.

The harmonisation procedure:

1. validates timestamps and removes exact duplicate records;
2. aggregates observations falling within the same second;
3. propagates the latest past observation for at most two seconds;
4. marks longer communication gaps as unavailable;
5. records whether each row is observed, short-held, or unavailable;
6. prevents future samples from being used for interpolation;
7. separates discontinuous telemetry segments.

Unavailable telemetry is not interpreted as physical standstill. Short-held rows do not count as genuine physical observations when constructing the degradation rule.

The exact 1-Hz representation refers to the harmonised analysis grid and does not imply that every original telemetry source was natively sampled at exactly 1 Hz.

## Operational onset definition

The operational state rule uses a 10-second past-only history.

A candidate degraded state requires:

* an active mission during at least 60% of the history;
* an external explanation during no more than 20% of the history;
* either:

  * stop share of at least 0.60 using a speed threshold of 0.03 m/s; or
  * mean speed below 0.055 m/s;
* Cartesian displacement below 0.03 m;
* at least four genuine physical observations;
* at least six seconds of genuine observation span.

The candidate state must persist for three seconds before an onset is registered. A new onset can be registered only after ten seconds of recovery.

For forecasting horizon `h`, a positive target indicates that the first confirmed operational onset occurs within `(t, t+h]` while the AGV is currently not in a confirmed degraded state and the mission remains at risk.

Forecast anchors are censored at:

* mission completion;
* an external hold;
* unavailable telemetry;
* the end of the mission leg;
* insufficient future follow-up.

Graph progress is not used in the target definition.

The target is an operational definition of ineffective mission execution. It is not an independently annotated mechanical-fault label.

## Route representation

If a valid controller-recorded route is available, it is retained.

When the route is unavailable, the pipeline reconstructs a minimum-distance directed path from the associated AGV position to the logged target node.

Route provenance is retained so that recorded and reconstructed route context are distinguishable. Reconstructed route variables are treated as approximate mission context rather than controller-recorded ground truth.

Graph-derived variables include:

* graph-route remaining distance;
* route completion;
* recent graph progress;
* graph progress rate;
* current-edge progress and remaining distance;
* graph-association variables;
* node and edge context;
* route availability and provenance.

Euclidean variables include straight-line target distance and recent Euclidean progress.

## Feature representations

Four controlled representations are evaluated:

1. **Base context**

   * physical telemetry;
   * recent speed and stopping behaviour;
   * electrical and wheel measurements;
   * command-consistency measurements;
   * observation age and availability;
   * missingness indicators;
   * route-source provenance.

2. **Base + Euclidean**

   * base context;
   * Euclidean target distance;
   * recent Euclidean progress.

3. **Base + Euclidean + graph**

   * base and Euclidean context;
   * directed-route distance and completion;
   * recent graph progress;
   * current-edge and node context;
   * graph-association variables.

4. **Base + Euclidean + graph, no SOC**

   * the full representation;
   * all SOC-derived variables removed.

Active-mission and external-hold variables are used for eligibility and censoring. They are not predictive model inputs.

## Models

The tabular evaluation includes:

* Logistic Regression;
* ExtraTrees with 400 trees;
* Random Forest with 350 trees;
* Histogram Gradient Boosting with 250 iterations.

All learned models use random seed 42. Class-balanced learning is used where supported.

A separate temporal baseline evaluates a single-layer GRU using:

* a fixed 10-second sequence;
* 32 hidden units;
* dropout of 0.15;
* class-weighted binary cross-entropy;
* AdamW;
* learning rate of `1e-3`;
* weight decay of `1e-4`;
* batch size of 128;
* 40 training epochs;
* random seed 42.

The GRU is included as a temporal baseline and is not presented as a novel neural architecture.

## Evaluation protocol

The primary protocol is complete-session leave-one-session-out (LOSO) evaluation.

For each fold:

1. one complete physical recording session is reserved for testing;
2. the other four sessions form the development data;
3. imputation, scaling, feature removal, and model fitting use development data only;
4. probability calibration excludes the test session;
5. the decision threshold is selected without using the test session.

When possible, a complete development session containing sufficient examples of both classes is reserved for Platt calibration. Otherwise, a class-valid chronological development tail is used.

The probability threshold is selected from 0.10 to 0.90 in increments of 0.02 using calibration-set macro-F1.

PR-AUC is the primary ranking metric because positive forecasting anchors are rare.

S2 and S5 do not contain operational onsets. PR-AUC is therefore undefined in those held-out folds and is averaged over the three positive-event sessions. Threshold-dependent metrics and Brier score are evaluated across all five sessions where applicable.

### Primary analysis and exploratory comparisons

The five-second Random Forest representation comparison is the primary reported representation analysis.

For completeness, the repository also reports the configuration with the highest observed mean PR-AUC at each forecasting horizon. Because these configurations are identified from the held-out LOSO results, the best-at-each-horizon comparison is treated as exploratory and descriptive rather than as an independently confirmed model-selection result.

## Primary five-second representation results

For Random Forest at the primary five-second forecasting horizon:

| Representation                   |   Mean PR-AUC |
| -------------------------------- | ------------: |
| Base context                     | 0.488 ± 0.190 |
| Base + Euclidean                 | 0.587 ± 0.174 |
| Base + Euclidean + graph         | 0.584 ± 0.259 |
| Base + Euclidean + graph, no SOC | 0.598 ± 0.267 |

PR-AUC is averaged across the three positive-event held-out sessions.

The Euclidean-plus-graph representation without SOC obtains mean PR-AUC 0.598, compared with 0.587 for the Euclidean representation and 0.488 for the common base context.

The improvement over Euclidean context is modest and varies across held-out sessions and models. The results do not support universal graph superiority.

## Exploratory across-horizon results

For descriptive comparison, the configuration with the highest observed mean PR-AUC at each forecasting horizon is:

| Horizon | Model and representation                  |        PR-AUC |      Macro-F1 |   Positive F1 |         Brier |
| ------: | ----------------------------------------- | ------------: | ------------: | ------------: | ------------: |
|     5 s | Random Forest / Euclidean + graph, no SOC | 0.598 ± 0.267 | 0.674 ± 0.172 | 0.356 ± 0.352 | 0.016 ± 0.015 |
|    10 s | Random Forest / Euclidean + graph, no SOC | 0.675 ± 0.145 | 0.563 ± 0.102 | 0.152 ± 0.212 | 0.038 ± 0.042 |
|    20 s | Histogram Gradient Boosting / Euclidean   | 0.597 ± 0.062 | 0.576 ± 0.102 | 0.211 ± 0.249 | 0.083 ± 0.081 |

PR-AUC is averaged across the three positive-event test sessions. The remaining metrics use all five held-out sessions.

These are exploratory highest-observed LOSO configurations and are not treated as independently selected confirmatory test configurations.

## Alert policy

A supervisory alert activates when at least two of the latest three calibrated probabilities exceed the development-selected threshold.

The policy also uses:

* a lower deactivation threshold;
* hysteresis;
* a five-second cooldown;
* state reset at session, telemetry-segment, mission-leg, or timestamp discontinuities.

An onset is counted as warned when the alert is active during its corresponding pre-onset horizon.

False-alert frequency is normalised by eligible monitored exposure rather than complete wall-clock recording duration.

## Alert replay results

At the five-second operating point, the Random Forest using Euclidean-plus-graph context without SOC:

* warns 45 of 67 operational onsets;
* achieves event-level recall of 0.672;
* produces 59 alert episodes;
* produces 15 false alert episodes;
* produces 7.50 false alert episodes per eligible forecasting hour;
* warns 34 events with at least two seconds of lead;
* warns 10 events with at least five seconds of lead;
* produces no alert episodes in S2 or S5.

The operating threshold is selected using development data only.

Session-level warned events are:

| Session | Warned / total |
| ------- | -------------: |
| S1      |        20 / 25 |
| S3      |         2 / 10 |
| S4      |        23 / 32 |

Performance is session-dependent, particularly in S3. The alert is therefore intended for supervisory or shadow-mode evaluation rather than autonomous control.

## Temporal GRU baseline

The fixed-sequence GRU obtains:

| Horizon |   Mean PR-AUC |
| ------: | ------------: |
|     5 s | 0.404 ± 0.187 |
|    10 s | 0.530 ± 0.254 |
|    20 s | 0.352 ± 0.212 |

At the primary five-second horizon, the GRU:

* warns 34 of 67 operational onsets;
* produces 27 false alert episodes;
* produces 13.49 false alerts per eligible forecasting hour.

At the primary five-second operating point, the GRU provides a less favourable ranking and alert trade-off than the reported Random Forest configuration.

## Communication-gap stress test

An ExtraTrees Euclidean-plus-graph model is evaluated after injecting communication gaps of 1, 2, 3, 5, and 10 seconds.

The aggregate values below use the `complete_test_session` rows in `outputs/scientific_experiments/05_gap_robustness.csv`, which is generated by `02_run_scientific_experiments.py`.

Prediction coverage is averaged across all five held-out sessions. PR-AUC is averaged across the three held-out sessions containing operational onsets.

| Injected gap | Mean prediction coverage | Mean PR-AUC |
| -----------: | -----------------------: | ----------: |
|          1 s |                    1.000 |       0.519 |
|          2 s |                    1.000 |       0.515 |
|          3 s |                    0.997 |       0.512 |
|          5 s |                    0.990 |       0.519 |
|         10 s |                    0.974 |       0.467 |

Longer gaps reduce prediction coverage after the two-second hold allowance. Ranking performance is comparatively stable through five-second perturbations in this experiment and decreases more clearly at ten seconds.

The stress test does not reproduce every possible communication delay, packet loss, reordering, or network failure.

## Latency

Across the evaluated estimators and folds, the largest measured per-row p99 pipeline prediction time in `outputs/scientific_experiments/06_latency.csv` is approximately:

```text
0.535 ms
```

The maximum committed value is approximately `0.5347568 ms/row`.

This measurement includes fitted in-pipeline preprocessing, transformation, and probability prediction.

It does not include:

* telemetry acquisition;
* graph-state construction;
* communication delay;
* alert delivery;
* TMS/MES processing;
* control execution.

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
├── 06_prepare_expert_audit.py
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
    ├── feature_availability.csv
    ├── session_audit.csv
    ├── label_definition.json
    ├── run_summary.json
    │
    ├── scientific_experiments/
    │   ├── 00_summary.json                    [generated by Step 2]
    │   ├── 01_exact_1hz_verification.csv     [generated by Step 2]
    │   ├── 02_multi_horizon_loso_ablation.csv
    │   ├── 03_predictions.csv                 [generated by Step 2]
    │   ├── 04_alert_policy_metrics.csv
    │   ├── 05_gap_robustness.csv              [generated by Step 2]
    │   ├── 06_latency.csv
    │   ├── 07_aggregate_results.csv
    │   ├── 08_representation_deltas.csv       [generated by Step 2]
    │   ├── 09_event_distribution.csv
    │   └── 10_experiment_manifest.json        [generated by Step 2]
    │
    ├── pre_onset_experiments/
    │   ├── 07_pre_onset_loso_results.csv
    │   ├── 08_pre_onset_summary.csv
    │   ├── 09_pre_onset_predictions.csv       [generated by Step 3]
    │   └── 10_pre_onset_best_models.csv       [generated by Step 3]
    │
    ├── provenance_hazard/
    │   ├── 00_manifest.json
    │   ├── 01_hazard_loso_results.csv
    │   ├── 02_hazard_predictions.csv
    │   ├── 03_hazard_coefficients.csv
    │   └── 04_hazard_aggregate.csv
    │
    ├── temporal_gru/
    │   ├── 00_manifest.json
    │   ├── 01_temporal_gru_loso_results.csv
    │   ├── 02_temporal_gru_summary.csv
    │   ├── 03_temporal_gru_predictions.csv
    │   ├── 04_temporal_alert_tradeoffs.csv
    │   └── 05_temporal_vs_best_tabular.csv
    │
    └── acceptance_audits/
        ├── 00_manifest.json
        ├── 01_alert_tradeoff_per_fold.csv
        ├── 02_alert_tradeoff_aggregate.csv
        ├── 03_expert_audit_key.csv
        ├── 04_expert_audit_form.csv
        └── 05_expert_audit_windows.csv
```

Some generated output files are not committed in the repository snapshot. They are recreated automatically when the corresponding experiment script is executed. Files marked as `[generated by Step 2]` are produced by `02_run_scientific_experiments.py`, while files marked as `[generated by Step 3]` are produced by `03_run_pre_onset_experiments.py`. These files are generated results and are not required as input data.

The provenance-conditioned hazard experiment is retained as an exploratory ablation and is not part of the primary claimed method.

The expert-audit script prepares review cases and forms. Running it does not itself constitute independent expert validation.

## Requirements

Python 3.10 or newer is recommended.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run order

### 1. Build the harmonised dataset

```bash
python 01_build_scientific_dataset.py
```

Primary output:

```text
outputs/harmonized_graph_mission_state.csv
```

### 2. Run the shared tabular experiments

```bash
python 02_run_scientific_experiments.py
```

This script generates the shared tabular LOSO outputs, prediction-level results, communication-gap stress test, latency benchmark, verification tables, and representation-comparison outputs. Generated files are written to `outputs/scientific_experiments/`.

### 3. Run the strict pre-onset evaluation

```bash
python 03_run_pre_onset_experiments.py
```

Important outputs:

```text
outputs/pre_onset_experiments/07_pre_onset_loso_results.csv
outputs/pre_onset_experiments/08_pre_onset_summary.csv
outputs/pre_onset_experiments/09_pre_onset_predictions.csv
outputs/pre_onset_experiments/10_pre_onset_best_models.csv
```

The prediction-level and highest-observed summary files are generated when this script is executed and do not need to be present before running the experiment.

The highest-observed configuration table is retained for exploratory across-horizon comparison.

### 4. Run the provenance-conditioned hazard ablation

```bash
python 04_run_provenance_hazard.py
```

This is an exploratory ablation.

### 5. Run the temporal GRU baseline

```bash
python 05_run_temporal_gru_baseline.py
```

PyTorch is required for this step.

### 6. Prepare expert-audit materials

```bash
python 06_prepare_expert_audit.py
```

This script prepares blinded cases and review forms. It does not itself provide an expert-validation result.

### 7. Generate the held-out warning figure

```bash
python 07_warning_figure.py
```

Input:

```text
outputs/pre_onset_experiments/09_pre_onset_predictions.csv
```

This input file is generated by Step 3. Run `03_run_pre_onset_experiments.py` before generating the warning figure.

Output:

```text
figures/final_successful_h5_warning.png
```
