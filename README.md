# Beyond Detection: Proactive Insider Threat Risk Forecasting

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CERT r4.2](https://img.shields.io/badge/Dataset-CERT%20r4.2-green.svg)](https://doi.org/10.1184/R1/12841247.v1)

## Overview
Proactive Insider Threat prediction using LSTM-based temporal Behaviour modelling with Early-Warning Lead Time (EWLT) evaluation.

**Key Results:**
| Metric        | LSTM   | Random Forest | Logistic Regression |     ofa       |   Other method      | 
|---------------|--------|---------------|---------------------|---------------|---------------------| 
| AUC-ROC       | 0.937  | 0.983         | 0.938               | 0.983         | 0.938               |
| Precision     | 0.135  | 0.079         | 0.027               | 0.983         | 0.938               |
| F1-Score      | 0.217  | 0.144         | 0.053               | 0.983         | 0.938               |
| FPR           | 0.0111 | 0.0291        | 0.1046              | 0.983         | 0.938               |
| **EWLT**      | **43 days** | N/A      | N/A                 | N/A           | N/A                 |
|---------------|--------|---------------|---------------------|---------------|---------------------| 
## Dataset
CERT Insider Threat Test Dataset r4.2 is **not redistributable**.
Request access at: https://doi.org/10.1184/R1/12841247.v1
Place raw files in `data/raw/` following `data/README_data.md`.

## Quick Start
```bash
git clone https://github.com/iun1xmd5/BeyDet.git
cd BeyDet
pip install -r requirements.txt

# 1. Preprocess
python src/preprocessing/feature_engineering.py

# 2. Train LSTM
python src/training/train_lstm.py --config configs/lstm_optimal.yaml

# 3. Evaluate + Compute EWLT
python src/evaluation/ewlt.py --model outputs/models/lstm_best.pt

# 4. Run Bayesian HPO (optional)
python src/training/bayesian_optimisation.py \
       --config configs/lstm_search_space.yaml

