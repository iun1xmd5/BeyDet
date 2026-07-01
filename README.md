# InsiderLSTM

Code for the paper **"Beyond Detection: A Proactive Insider Threat Risk
Forecasting using LSTM-Based Temporal Behavioural Modelling and Early-Warning
Lead Time Evaluation."**

InsiderLSTM is a unidirectional LSTM with per-feature input gating and highway
state refinement, trained on daily-aggregated user behaviour sequences (42
features: 21 raw counts + 21 rolling z-score deviations) over a 14-day causal
window. Alongside standard classification metrics it reports **Early-Warning
Lead Time (EWLT)** — the number of days before the first confirmed malicious
event at which the model raises a sustained above-threshold risk signal.

Evaluated on **CERT r4.2** and **SPEDIA** under scenario-stratified temporal
splits: AUC-ROC 0.9761 / F1 0.759 / mean EWLT 8.3 days on CERT r4.2, and
AUC-ROC 0.9805 / F1 0.968 / mean EWLT 2.0 days on SPEDIA.

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
construction, daily feature aggregation, rolling z-score features,
scenario-stratified temporal splitting, 14-day sequence construction, model
training with Bayesian hyperparameter optimisation, baseline/SOTA comparisons,
EWLT analysis, statistical significance tests, and the ablation study.

Paths resolve automatically for Kaggle, Colab, or a local run. For a local run
with data stored outside the repo:

    DATA_DIR=/path/to/data python src/main.py

Generated artifacts (parquet, json, csv, figures, `.pth` checkpoints) are
written to `./outputs` by default; override with `WORK_DIR`. Sections that
reload artifacts from a previous run read from `RELOAD_DIR` (defaults to
`WORK_DIR`).

## Notebook vs. script

`src/main.py` and `notebooks/r4-2-and-spedia-final.ipynb` contain the same
pipeline. The notebook is kept for interactive/Kaggle use; the script is the
convenient entry point for a plain `python` run. Both share the identical
corrected path-resolution block.

## Citation

    @article{ndewingia_insiderlstm,
      title   = {Beyond Detection: A Proactive Insider Threat Risk Forecasting
                 using LSTM-Based Temporal Behavioural Modelling and
                 Early-Warning Lead Time Evaluation},
      author  = {Ndewingia, Hillary Gabriel and others},
      year    = {2026}
    }

Update the citation block with the final author list and venue once available.
