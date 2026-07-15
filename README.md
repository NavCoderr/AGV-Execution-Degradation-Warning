# Mission-Aware Early Warning of AGV Execution Degradation

This repository contains the implementation and experimental results for a mission-aware supervisory monitoring framework for graph-based automated guided vehicle (AGV) operation.

The method identifies situations in which an AGV still has an active destination command but is no longer making effective mission progress. It distinguishes unexpected progress degradation from stops explained by operator intervention, scanner events, deadlock handling, emergency holding, or traffic-management actions.

The framework combines recent motion, commanded-motion consistency, power and SOC context, target progress, and graph-node/edge information to estimate short-horizon mission-execution degradation from real physical-AGV telemetry.

The output is intended for supervisory monitoring and decision support. It is not a certified safety controller or a battery-health estimator.

## Dataset

The evaluation uses two separately recorded sessions from the same physical AGV.

| Session | Raw observations | SOC range |
|---|---:|---:|
| Low-SOC stress | 4,240 | 59% to 2% |
| High-SOC control | 790 | 83% to 79% |

The laboratory navigation graph contains 35 nodes and 102 directed edges.

At the 5 s warning horizon, the processing pipeline produces:

| Metric | Low-SOC | High-SOC |
|---|---:|---:|
| Gap-aware aligned timestamps | 7,681 | 1,591 |
| Eligible labelled timestamps | 3,521 | 379 |
| Mission legs | 32 | 6 |
| Positive-label rate | 0.298 | 0.016 |

Aligned timestamps are used to construct temporal windows and are not treated as independent experimental repetitions.

## Method

The pipeline:

1. aligns each telemetry session to a 1 Hz timeline;
2. limits forward filling to two seconds and separates longer communication gaps;
3. identifies active destination execution;
4. excludes externally explained stops and holds;
5. associates each eligible state with graph node, edge, target, and remaining distance;
6. constructs a 30 s history of motion, command mismatch, power, SOC, target progress, and graph context;
7. predicts degradation over warning horizons of 0 s, 5 s, 10 s, and 20 s.

A positive label represents future loss of effective mission progress during active execution. Labels are derived from future commanded-motion mismatch, stop share, low speed, and insufficient target progress.

## Evaluation

The evaluated estimators are:

- Logistic Regression;
- ExtraTrees;
- Random Forest.

The primary protocol uses five-fold mission-leg grouped cross-validation. All timestamps from the same mission leg remain entirely within either training or testing.

The high-SOC session is used as a separate operating-condition control after training on the low-SOC session.

## Main results

ExtraTrees achieved the strongest mean macro-F1:

| Warning horizon | Macro-F1 |
|---:|---:|
| 0 s | 0.910 ± 0.057 |
| 5 s | 0.878 ± 0.035 |
| 10 s | 0.812 ± 0.036 |
| 20 s | 0.719 ± 0.031 |

At the 5 s horizon:

- macro-F1: `0.878 ± 0.035`;
- PR-AUC: `0.832 ± 0.040`;
- ROC-AUC: `0.916 ± 0.044`;
- positive-class recall: `0.865 ± 0.089`;
- 16 of 18 positive low-SOC mission legs were warned;
- 3 of 29 low-SOC mission legs produced false warnings;
- median first-warning lead time was `0.5 s`.

The 5 s horizon defines the beginning of the future label window. It does not imply that every warning provides five seconds of actionable lead time.

## Ablation summary

The SOC-only configuration achieved `0.675 ± 0.118` macro-F1, compared with `0.878 ± 0.035` for the full mission state.

This indicates that SOC alone is insufficient. The main warning evidence comes from realised motion and effective mission progress, with graph context providing complementary mission-specific information.

## Repository contents

```text
README.md
requirements.txt
icra_final_experiments.py

01_real_agv_live_LOW_SOC_STRESS.csv
02_real_agv_live_HIGH_SOC_CONTROL.csv
Node_F3.csv
Edge_Distances3.csv

icra_final_results/
```

The result folder contains evaluation tables, mission-level metrics, ablation results, threshold analysis, and representative case figures.

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python experiments.py \
  --data_dir . \
  --out_dir icra_final_results
```

## Scope

This repository reports a single-platform, two-session feasibility study.

The current results do not establish fleet-level generalisation, causal battery-failure prediction, certified safety performance, route replanning, or closed-loop traffic-management intervention.
