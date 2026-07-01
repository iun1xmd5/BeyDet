**"Beyond Detection: A Proactive Insider Threat Risk
Forecasting using LSTM-Based Temporal Behavioural Modelling and Early-Warning
Lead Time Evaluation."**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CERT r4.2](https://img.shields.io/badge/Dataset-CERT%20r4.2-green.svg)](https://doi.org/10.1184/R1/12841247.v1)

# Overview

InsiderLSTM is a unidirectional LSTM with per-feature input gating and highway
state refinement, trained on daily-aggregated user behaviour sequences (42
features: 21 raw counts + 21 rolling z-score deviations) over a 14-day causal
window. Alongside the standard classification metrics it reports **Early-Warning
Lead Time (EWLT)** — the number of days before the first confirmed malicious
event at which the model raises a sustained above-threshold risk signal.

Evaluated on **CERT r4.2** and **SPEDIA** under scenario-stratified temporal
splits:

- **CERT r4.2:** AUC-ROC 0.9841, F1 0.740, FPR 0.006, mean EWLT 5.0 days across
  five detected insiders (7.1% coverage of the 70-insider test set).
- **SPEDIA:** AUC-ROC 0.9805, F1 0.968, mean EWLT 2.0 days from a single
  detected insider.

DeLong tests (Bonferroni α* = 0.0083) confirm significant AUC-ROC advantages
over CNN-GRU, OFA-LSTM, and TA-LSTM on CERT r4.2; the gap with ITDSTS is
negligible (ΔAUC = 0.0003) and the two are statistically equivalent. On SPEDIA
the near-balanced classes compress the AUC range and only CNN-GRU and OFA-LSTM
reach significance.

## Repository layout

    .
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── src/
    │   └── main.py         # full pipeline, converted from the notebook
    ├── notebooks/
    │   └── r4-2-and-spedia-final.ipynb   # original notebook (paths corrected)
    └── data/
        └── README.md       # dataset sources and expected layout

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Experiments in the paper used Python 3.10, PyTorch 2.4, and dual NVIDIA T4 GPUs
on Kaggle. A CUDA GPU is strongly recommended; the code falls back to CPU.

## Data

The datasets are **not** included. See [`data/README.md`](data/README.md) for
download links (CERT r4.2 from CMU KiltHub, SPEDIA from Zenodo) and the exact
folder layout the code expects.

## Running

    python src/main.py

`src/main.py` runs the full pipeline top to bottom: log loading and label
construction, daily feature aggregation across five activity sources, rolling
z-score deviation features, scenario-stratified temporal splitting,
14-day sequence construction, model training with Bayesian hyperparameter
optimisation, baseline and state-of-the-art comparisons, EWLT analysis,
DeLong/Wilcoxon significance tests, and the ablation study.

Paths resolve automatically for Kaggle, Colab, or a local run. For a local run
with data stored outside the repo:

    DATA_DIR=/path/to/data python src/main.py

Generated artifacts (parquet, json, csv, figures, `.pth` checkpoints) are
written to `./outputs` by default; override with `WORK_DIR`. Sections that
reload artifacts from a previous run read from `RELOAD_DIR` (defaults to
`WORK_DIR`).

## Method summary

- **Features:** 21 raw daily behavioural features from five sources (logon,
  device, file, email, HTTP) plus 21 rolling z-score deviations computed over a
  14-day window with a one-day shift to prevent temporal leakage; standardised
  with a StandardScaler fitted on the training partition only.
- **Model:** per-feature sigmoid input gate → stacked unidirectional LSTM →
  highway-style state refinement → batch-norm/dropout → two-layer classifier.
- **Training:** weighted BCE with `pos_weight` set dynamically to the square
  root of the negative-to-positive sequence ratio (5.97 on CERT r4.2, 1.00 on
  SPEDIA); Adam with weight decay 1e-4; CosineAnnealingLR; early stopping on
  validation AUC (patience 25).
- **Hyperparameter search:** Bayesian optimisation (scikit-optimize, GP + EI)
  over 45 trials in two passes; the top five candidates by validation AUC are
  retrained on the full training set and the final config is chosen by test
  AUC-ROC. This makes the final selection partially test-set-informed — noted
  as a limitation in the paper (Section 6.4).
- **EWLT:** two-day sustained crossing of a separately calibrated threshold
  (θ_EWLT = 0.60), searched strictly before the first confirmed incident day.

## Notebook vs. script

`src/main.py` and `notebooks/r4-2-and-spedia-final.ipynb` contain the same
pipeline. The notebook is kept for interactive/Kaggle use; the script is the
convenient entry point for a plain `python` run. Both share the identical
corrected path-resolution block.

citation block with the final author list and venue once available.
