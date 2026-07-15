# Mission-Aware Early Warning of AGV Execution Degradation

This repository contains the experiment pipeline and supporting results for a mission-aware supervisory monitoring framework for graph-based automated guided vehicle (AGV) operation.

The framework addresses a runtime condition in which an AGV still has an active destination command but is no longer executing the assigned mission effectively. A stopped or slow-moving AGV is not automatically considered degraded because similar telemetry may also be produced by station waiting, operator intervention, scanner stops, deadlock handling, emergency holding, braking, or traffic-management commands.

The implemented workflow first identifies valid active-mission execution, excludes observations explained by recorded external holds, links the physical AGV state to the current graph node, directed edge, and assigned target, and then estimates whether effective mission progress will be lost within a short future horizon.

The output is intended for supervisory traffic-management or operator support. It is not a certified safety controller, a battery-health estimator, or a replacement for low-level AGV safety functions.

## Experiment

The evaluation uses two separately recorded telemetry sessions from the same physical AGV:

- a low-state-of-charge stress session in which SOC decreases from 59% to 2%;
- a high-state-of-charge control session in which SOC remains between 83% and 79%.

The AGV operates on a laboratory navigation graph containing:

- 35 nodes;
- 102 directed edges.

The source logs contain 5,030 raw physical-AGV observations:

| Session | Raw observations | SOC range |
|---|---:|---:|
| Low-SOC stress | 4,240 | 59% to 2% |
| High-SOC control | 790 | 83% to 79% |

At the 5 s warning horizon, the gap-aware alignment and mission filtering produce:

| Metric | Low-SOC | High-SOC |
|---|---:|---:|
| Gap-aware aligned timestamps | 7,681 | 1,591 |
| Active-mission timestamps | 4,394 | 679 |
| Primary eligible timestamps | 3,649 | 391 |
| Eligible labelled timestamps | 3,521 | 379 |
| Mission legs | 32 | 6 |
| Positive-label rate | 0.298 | 0.016 |

Aligned timestamps are used to construct uniformly spaced temporal windows. Forward-filled timestamps are not treated as new independent physical observations or additional experimental trials.

## Monitoring workflow

The experiment performs the following operations:

1. Loads the low-SOC and high-SOC physical-AGV telemetry sessions.

2. Loads the graph-node and directed-edge definitions.

3. Aligns each telemetry session independently to a 1 Hz timeline.

4. Limits forward filling to at most two consecutive seconds.

5. Separates longer communication gaps into independent temporal segments.

6. Identifies timestamps corresponding to active destination execution.

7. Excludes timestamps explicitly associated with:
   - idle or missing-goal operation;
   - operator intervention;
   - scanner-triggered stopping;
   - deadlock handling;
   - braking or emergency holding;
   - traffic-management holding or yielding.

8. Links each eligible AGV state to:
   - the snapped graph node;
   - the current directed edge;
   - edge validity and edge distance;
   - node degree and node type;
   - the assigned target node;
   - distance to the target.

9. Constructs a 30 s mission-execution history using:
   - requested and measured motion;
   - stop behaviour;
   - commanded-motion mismatch;
   - physical displacement;
   - power and voltage behaviour;
   - SOC context;
   - target progress;
   - graph-mission context.

10. Builds future mission-execution labels using a 10 s future observation window.

11. Evaluates warning horizons of:
    - 0 s;
    - 5 s;
    - 10 s;
    - 20 s.

12. Uses mission-leg grouped cross-validation so that timestamps from the same mission leg cannot appear in both training and test partitions.

13. Performs:
    - estimator comparison;
    - mission-state ablation;
    - threshold sensitivity analysis;
    - high-SOC control assessment;
    - mission-level warning analysis;
    - true-positive, false-positive, and false-negative case inspection.

## Mission-execution target

For each eligible timestamp, the target is calculated from a 10 s future observation window beginning after the selected warning horizon.

A positive mission-execution label is assigned when the future interval remains predominantly within active destination execution and at least one of the following conditions is satisfied:

- commanded motion is present while the AGV remains stopped for at least half of the future window;
- future stop share is at least 0.60 and target progress is below 0.03 m;
- mean future speed is below 0.055 m/s and target progress is below 0.03 m.

These labels represent operational loss of active-mission progress. They are not manually annotated mechanical-failure labels and do not establish a causal relationship between low SOC and AGV failure.

## Warning estimators

The constructed mission-execution state is evaluated using three established estimators:

- Logistic Regression;
- Extremely Randomized Trees, reported as ExtraTrees;
- Random Forest.

The estimators are used to test whether the mission-aware state supports reliable warning across linear and nonlinear decision boundaries.

The experiment uses:

- training-fold median imputation for numerical variables;
- training-fold mode imputation for categorical variables;
- one-hot encoding with unseen categories ignored;
- numerical standardization;
- balanced class weights;
- 500 trees for ExtraTrees and Random Forest;
- minimum leaf size of three;
- random seed 42.

## Evaluation protocol

Random row-wise splitting is not used because neighbouring timestamps from the same mission execution are strongly correlated.

The primary evaluation uses five-fold mission-leg grouped cross-validation. All timestamps from one mission leg remain entirely in either the training or test partition.

The low-SOC stress session is used for grouped evaluation. After training on all eligible low-SOC mission legs, the high-SOC session is evaluated as an untouched operating-condition control.

Because the high-SOC session contains very few positive-label intervals, it is used mainly to assess conservative false-warning behaviour. It is not treated as evidence of fleet-level, cross-platform, or broad cross-condition generalisation.

The reported metrics include:

- macro-averaged F1;
- positive-class precision;
- positive-class recall;
- positive-class F1;
- ROC-AUC;
- PR-AUC;
- Brier score;
- mission-level warning recall;
- false-warning mission legs;
- false warnings per mission leg;
- first-warning lead time.

## Main results

ExtraTrees achieved the strongest mean macro-F1 at all evaluated horizons.

| Warning horizon | Macro-F1 |
|---:|---:|
| 0 s | 0.910 ± 0.057 |
| 5 s | 0.878 ± 0.035 |
| 10 s | 0.812 ± 0.036 |
| 20 s | 0.719 ± 0.031 |

At the primary 5 s horizon:

- macro-F1: `0.878 ± 0.035`;
- PR-AUC: `0.832 ± 0.040`;
- ROC-AUC: `0.916 ± 0.044`;
- positive-class recall: `0.865 ± 0.089`.

Mission-level results at the 5 s horizon and a threshold of 0.50 are:

| Metric | Low-SOC grouped CV | High-SOC control |
|---|---:|---:|
| Evaluated mission legs | 29 | 6 |
| Mission legs with positive labels | 18 | 3 |
| Warned positive mission legs | 16 | 0 |
| Mission-level warning recall | 0.889 | 0.000 |
| False-warning mission legs | 3 | 0 |
| False warnings per mission leg | 0.103 | 0.000 |
| Median first-warning lead | 0.5 s | — |

The 5 s horizon indicates where the future label window begins. It does not imply that every warned mission receives five seconds of actionable lead time.

## Mission-state ablation

The ablation study shows that the monitoring decision cannot be reduced to a battery or SOC threshold.

| State configuration | Macro-F1 |
|---|---:|
| No target progress | 0.883 ± 0.031 |
| No historical command mismatch | 0.880 ± 0.038 |
| Full mission state | 0.878 ± 0.035 |
| No SOC/voltage | 0.869 ± 0.037 |
| No graph context | 0.862 ± 0.056 |
| Motion + graph | 0.854 ± 0.048 |
| Motion only | 0.841 ± 0.045 |
| Motion + progress | 0.838 ± 0.044 |
| SOC only | 0.675 ± 0.118 |

The principal warning evidence is associated with realised motion, commanded-motion consistency, and effective mission progress. Graph context provides complementary information about where that behaviour occurs within the assigned mission. SOC describes the operating condition but is not sufficient as the warning decision itself.

## Run the experiment

Install the required packages:

```bash
pip install -r requirements.txt
