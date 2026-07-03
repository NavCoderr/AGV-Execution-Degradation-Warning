# AGV Execution Degradation Warning

This repository contains code and supporting material for an early-warning framework for execution degradation in graph-based automated guided vehicle (AGV) missions under low state-of-charge operation.

The framework uses real AGV telemetry, graph-context features, motion-stability indicators, power signals, target-progress features, and supervised machine-learning models to estimate future execution degradation. The predicted warning probability is converted into a supervisory advisory for traffic-management or operator support.

## Main components

- 1 Hz telemetry synchronization
- graph/node-edge alignment
- past-window feature extraction
- future-horizon degradation labeling
- supervised warning models
- prediction-horizon evaluation
- feature-group ablation
- offline advisory replay
- case-study visualization

## Dataset

The experiments use two physical AGV telemetry sessions:

- low-SOC stress session
- high-SOC control session

The processed dataset contains 9706 synchronized 1 Hz samples mapped to a graph with 35 nodes and 102 directed edges.

Raw laboratory telemetry may be restricted. Anonymized processed features or result tables can be provided where permitted.

## Models

The evaluated models include:

- Logistic Regression
- ExtraTrees
- Random Forest
- Gradient Boosting

## Results

The best model achieved:

- 0.778 macro-F1 for 5 s early warning
- 0.754 macro-F1 for 10 s early warning

## Status

This repository is under preparation for research reproducibility.
