# Datasets

The datasets used in this study are **not redistributed** in this repository
due to size and licensing. Download them from the original sources below and
place the files in this folder as described. Nothing under `data/` is tracked
by git except this README (see `.gitignore`).

The paper uses two datasets: **CERT r4.2** and **SPEDIA**.

## 1. CERT r4.2 (CMU Insider Threat Test Dataset)

Synthetic insider-threat benchmark from Carnegie Mellon University: 1,000 users
over ~17 months, with 70 confirmed insiders across three threat scenarios.

- Source (all releases live under one record): https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247
- DOI: 10.1184/R1/12841247.v1 (Lindauer, 2020)
- Background paper: Glasser & Lindauer (2013), "Bridging the Gap: A Pragmatic
  Approach to Generating Insider Threat Data", doi:10.1109/SPW.2013.37
- Download `r4.2.tar.bz2`, decompress it, and place the five activity log files
  plus the `answers/` folder as shown below.

Files the pipeline reads: `logon.csv`, `device.csv`, `file.csv`, `email.csv`,
`http.csv`, and the per-scenario ground-truth files under `answers/`.

## 2. SPEDIA (System for Prediction and Early Detection of Insider Attacks)

Hybrid dataset combining real logs from 17 participants (24-day exercise,
March 2025) with simulated benign activity and CERT-derived augmentation.
Near-balanced (~57.5% malicious), with MITRE ATT&CK technique labels.

- Source: https://doi.org/10.5281/zenodo.15495572
- DOI: 10.5281/ZENODO.15495572 (Alvarez et al., 2025)
- Companion paper: Alvarez Muniz et al. (2026), "Design and generation of a
  dataset for training insider threat prevention and detection models: The
  SPEDIA dataset", Computers & Security, doi:10.1016/j.cose.2025.104743
- The annotated English CSV is expected at:
  `data/spedia/logs_SPEDIA_annotated_en.csv`

## Expected folder layout after download

    data/
    ├── r4.2/
    │   ├── logon.csv
    │   ├── device.csv
    │   ├── file.csv
    │   ├── email.csv
    │   └── http.csv
    ├── answers/
    │   ├── insiders.csv
    │   ├── r4.2-1/           # scenario 1 ground-truth files
    │   ├── r4.2-2/
    │   └── r4.2-3/
    └── spedia/
        └── logs_SPEDIA_annotated_en.csv

## Path resolution

`src/main.py` auto-detects the environment:

- **Kaggle** — reads from `/kaggle/input/...`
- **Colab** — reads from `/content/drive/MyDrive/CERT/...`
- **Local / repo** — reads from `./data` (this folder)

To point at data stored elsewhere on a local run, set an environment variable:

    DATA_DIR=/path/to/my/data python src/main.py

If your `answers/` subfolder names differ, adjust `ANS_DIR` handling in the
config block near the top of `src/main.py`.
