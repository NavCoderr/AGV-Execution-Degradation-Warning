# AGV Execution Degradation Warning

This repository contains the experiment files for an early-warning framework for execution degradation in graph-based automated guided vehicle (AGV) missions under low state-of-charge operation.

The goal is to monitor a running AGV mission using real telemetry and graph context, estimate whether the execution is degraded or likely to become degraded in a short future horizon, and convert the warning probability into a supervisory advisory for traffic-management or operator support.

## What this experiment does

The workflow is:

1. Load two real AGV telemetry sessions:
   - a low-SOC stress session
   - a high-SOC control session

2. Load the AGV graph files:
   - graph nodes
   - directed edge distances

3. Synchronize telemetry to 1 Hz.

4. Align each AGV sample with graph/node-edge context.

5. Extract past-window features from:
   - SOC dynamics
   - motion behaviour
   - power signals
   - target progress
   - graph context

6. Build future-horizon degradation targets using future speed, stop share, target progress, and brake evidence.

7. Train and evaluate supervised warning models.

8. Compare prediction horizons at 0 s, 5 s, 10 s, and 20 s.

9. Run feature-group ablation.

10. Generate advisory-replay summaries and representative case-study plots.

## Input files

The main input files are:

- `01_real_agv_live_LOW_SOC_STRESS.csv`  
  Real AGV telemetry session where SOC decreases from 59% to 2%.

- `02_real_agv_live_HIGH_SOC_CONTROL.csv`  
  Real AGV telemetry session where SOC remains between 83% and 79%.

- `Edge_Distances3.csv`  
  Directed edge-distance information for the AGV graph.

- `Node_F3.csv`  
  Node information used for graph alignment.

## Scripts

- `low_soc_experiment.py`  
  Runs the main experiment: preprocessing, feature extraction, label construction, model training, evaluation, ablation, advisory replay, and result export.

- `case_plots.py`  
  Generates representative advisory-replay case-study plots.

## Output folder

The folder `low_soc_real_two_session_results/` contains compact result summaries and generated outputs, including:

- `experiment_summary.json`
- `model_comparison.csv`
- `feature_ablation.csv`
- `session_control_advisory_summary.csv`
- `degradation_metrics_aggregated.csv`
- `degradation_metrics_blocked_split.csv`
- `feature_importance.csv`
- `label_sensitivity.csv`
- generated plots and case-study figures

Large intermediate prediction files may be omitted from the repository to keep the upload size manageable.

## Dataset summary

The experiment uses two physical AGV sessions:

| Session | Role | Raw rows | SOC range |
|---|---:|---:|---:|
| S1 | Low-SOC stress | 4240 | 59% to 2% |
| S2 | High-SOC control | 790 | 83% to 79% |

The processed dataset contains 9706 synchronized 1 Hz samples mapped to a graph with 35 nodes and 102 directed edges.

After history-window and horizon-specific filtering, the number of evaluated samples differs slightly across prediction horizons.

## Models

The evaluated supervised models are:

- Logistic Regression
- ExtraTrees
- Random Forest
- Gradient Boosting

The evaluation uses a blocked temporal split to avoid random mixing of neighbouring time samples.

## Main results

The best warning model achieved:

| Horizon | Macro-F1 |
|---:|---:|
| 5 s | 0.778 |
| 10 s | 0.754 |

The advisory replay showed that the low-SOC stress session produced higher warning probability, higher high-risk share, and non-zero charge-check recommendations compared with the high-SOC control session.

## Run

Install dependencies:

```bash
pip install -r requirements.txt

```

Run the main experiment:

```bash
python low_soc_experiment.py
```

Generate case-study plots:

```bash
python case_plots.py
```
