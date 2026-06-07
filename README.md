# Fraud Detection System

A machine learning system for detecting fraudulent transactions across two
distinct data pipelines: e-commerce behavioral data and anonymized bank
credit card transactions.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [File Organization Rationale](#file-organization-rationale)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Pipeline Summary](#pipeline-summary)
- [Key Design Decisions](#key-design-decisions)
- [Data Sources](#data-sources)


## Project Overview

Fraud detection is a high-stakes classification problem with two defining
characteristics that make it different from typical ML tasks:

1. **Severe class imbalance** fraud accounts for less than 10% of
   transactions in most real-world datasets, sometimes as low as 0.17%.
   Standard accuracy metrics are misleading; we use AUC-PR and F1-score.

2. **Asymmetric costs** a missed fraud (false negative) costs the business
   money and damages customer trust. A false positive (blocking a legitimate
   transaction) also damages customer trust. The model must balance both.

This project builds and evaluates fraud detection models for two separate
datasets, each with its own preprocessing pipeline, feature engineering
strategy, and modeling approach.


## Dataset Description

| Dataset | Rows | Features | Target | Challenge |
|---|---|---|---|---|
| `Fraud_Data.csv` | ~150,000 | Behavioral: user, device, IP, browser, timestamps | `class` (0/1) | Rich features needing engineering + geolocation |
| `IpAddress_to_Country.csv` | ~138,000 | IP integer ranges → country | — | Range-based merge, not direct join |
| `creditcard.csv` | ~284,807 | V1–V28 (PCA), Time, Amount | `Class` (0/1) | Extreme imbalance (0.17% fraud), anonymized features |

### Fraud_Data.csv: Column Reference

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Unique user identifier |
| `signup_time` | datetime | Account creation timestamp |
| `purchase_time` | datetime | Transaction timestamp |
| `purchase_value` | float | Transaction amount ($) |
| `device_id` | string | Device identifier |
| `source` | categorical | Traffic source (SEO, Ads, Direct) |
| `browser` | categorical | Browser used |
| `sex` | categorical | User gender |
| `age` | int | User age |
| `ip_address` | float | IP address stored as 32-bit integer float |
| `class` | int | Target — 1 = fraud, 0 = legitimate |

### creditcard.csv: Column Reference

| Column | Type | Description |
|---|---|---|
| `Time` | float | Seconds elapsed since first transaction |
| `V1`–`V28` | float | PCA-transformed features (anonymized) |
| `Amount` | float | Transaction amount |
| `Class` | int | Target — 1 = fraud, 0 = legitimate |


## Project Structure

fraud-detection/
├── .github/
│   └── workflows/
│       └── unittests.yml        # CI pipeline runs on push to main/task-*
├── data/
│   ├── raw/                     # Original datasets (gitignored)
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   └── processed/               # Cleaned, model-ready datasets (gitignored)
│       ├── fraud_train.csv
│       ├── fraud_test.csv
│       ├── fraud_featured.csv
│       ├── creditcard_train.csv
│       └── creditcard_test.csv
├── notebooks/
│   ├── eda-fraud-data.ipynb     # Full pipeline for Fraud_Data
│   ├── eda-creditcard.ipynb     # Full pipeline for creditcard
│   ├── feature-engineering.ipynb# Standalone feature engineering docs
│   ├── modeling.ipynb           # Task 2 model training
│   ├── shap-explainability.ipynb# Task 3 SHAP analysis
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load raw CSVs
│   ├── data_cleaner.py          # Clean, deduplicate, fix dtypes
│   ├── eda.py                   # All plotting and analysis functions
│   ├── geolocation.py           # IP → country enrichment
│   ├── feature_engineering.py   # Create new features for Fraud_Data
│   ├── transformation.py        # Scale, encode, split
│   └── imbalance_handler.py     # SMOTE, undersampling, SMOTETomek
├── tests/
│   ├── __init__.py
│   └── test_placeholder.py      # Growing test suite
├── models/                      # Saved model artifacts (gitignored)
│   ├── fraud_scaler.pkl
│   ├── fraud_encoder.pkl
│   └── creditcard_scaler.pkl
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md

## File Organization Rationale

### Why Feature Engineering Lives Inside the EDA Notebooks

A common question is why `feature_engineering.py` is called inside
`eda-fraud-data.ipynb` rather than in a standalone `feature-engineering.ipynb`.

**The reason is observational dependency.**

Feature engineering and EDA are not independent steps, they inform each
other in a continuous loop:

1. **EDA reveals the signal, feature engineering captures it.**
   You cannot decide to create `time_since_signup` without first observing
   in the EDA that fraudsters sign up and transact within minutes. The
   feature is a direct response to the EDA finding. Separating them would
   break that narrative thread.

2. **Engineered features need to be immediately analyzed.**
   After creating `hour_of_day`, the next natural step is to plot fraud rate
   by hour and observe whether there is a pattern. That observation belongs
   in the same notebook, right after the feature is created, not in a
   separate file that the reader has to cross-reference.

3. **The EDA notebook is the story of the data.**
   A reviewer or stakeholder reading `eda-fraud-data.ipynb` should see
   the complete journey: raw data → findings → features created in response
   to findings → features validated visually → data ready for modeling.
   Splitting this across two notebooks fragments the story.

**What `feature-engineering.ipynb` is for:**
The standalone `feature-engineering.ipynb` serves a different purpose — it
is a reference document that explains *what* each feature is, *why* it was
chosen, its mathematical definition, and its observed discriminating power.
It does not run the pipeline. Think of it as the documentation notebook,
while `eda-fraud-data.ipynb` is the execution notebook.

**Why this does not apply to creditcard:**
`eda-creditcard.ipynb` has no feature engineering section because the
creditcard dataset's features (V1–V28) are already PCA-extracted by the
issuing bank. There is nothing to engineer only to analyze and transform.


## Setup Instructions

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Datasets

Download the three datasets and place them in `data/raw/`:

data/raw/
├── Fraud_Data.csv
├── IpAddress_to_Country.csv
└── creditcard.csv

> These files are gitignored and must be obtained separately.
> See [Data Sources](#data-sources) for download links.

### 5. Create Required Directories

```bash
mkdir -p data/processed models
```


## How to Run

### Option A VS Code Notebooks

1. Open the project folder in VS Code
2. Install the Jupyter extension if not already installed
3. Open `notebooks/eda-fraud-data.ipynb`
4. Select your `.venv` Python kernel
5. Run all cells top to bottom (`Ctrl+Shift+P` → `Run All Cells`)
6. Repeat for `notebooks/eda-creditcard.ipynb`

### Option B Jupyter in Browser

```bash
jupyter notebook notebooks/
```

### Run Order

Always run the EDA notebooks before modeling:

1. notebooks/eda-fraud-data.ipynb     ← produces fraud_train/test.csv
2. notebooks/eda-creditcard.ipynb     ← produces creditcard_train/test.csv
3. notebooks/feature-engineering.ipynb← reference / documentation
4. notebooks/modeling.ipynb           ← Task 2 (requires outputs from 1 & 2)
5. notebooks/shap-explainability.ipynb← Task 3 (requires outputs from 4)

### Run Tests

```bash
pytest tests/ -v
```

## Pipeline Summary

### Fraud_Data Pipeline

Raw Data (Fraud_Data.csv + IpAddress_to_Country.csv)
        ↓
Data Cleaning
  - Remove duplicates
  - Fix dtypes: signup_time/purchase_time → datetime, class → int
  - Handle missing values (strategy documented per column)
        ↓
Exploratory Data Analysis
  - Class imbalance quantification
  - Univariate distributions: purchase_value, age
  - Categorical distributions: source, browser, sex
  - Bivariate analysis vs fraud class
  - Correlation heatmap
        ↓
Geolocation Integration
  - Convert float IPs to integers (732758368.7997 → 732758368)
  - Binary search range-merge with IpAddress_to_Country.csv
  - Fraud rate analysis by country
        ↓
Feature Engineering
  - hour_of_day, day_of_week (from purchase_time)
  - time_since_signup_hours (purchase_time − signup_time)
  - transaction_count_1h, transaction_count_24h (per user rolling window)
  - transaction_velocity (count_24h / account_age_hours)
        ↓
Data Transformation
  - Drop: user_id, device_id, ip_address, raw timestamps
  - Train/test split: 80/20 stratified
  - OneHotEncode: source, browser, sex, country (fit on train only)
  - StandardScale: numeric features (fit on train only)
        ↓
Imbalance Handling (training set only)
  - Technique: SMOTETomek
  - Justification: overlapping distributions → Tomek cleans boundary
        ↓
Output
  - data/processed/fraud_train.csv
  - data/processed/fraud_test.csv
  - models/fraud_scaler.pkl
  - models/fraud_encoder.pkl

### creditcard Pipeline

Raw Data (creditcard.csv)
        ↓
Data Cleaning
  - Remove duplicates
  - Ensure Class is int
        ↓
Exploratory Data Analysis
  - Extreme class imbalance (0.17% fraud)
  - Time and Amount distributions
  - PCA component analysis (V1–V28)
  - Top features correlated with Class
  - Correlation heatmap
        ↓
Data Transformation
  - Train/test split: 80/20 stratified
  - StandardScale: Time and Amount only
    (V1–V28 already scaled by PCA)
        ↓
Imbalance Handling (training set only)
  - Technique: SMOTETomek
  - Justification: 578:1 ratio makes undersampling impractical
        ↓
Output
  - data/processed/creditcard_train.csv
  - data/processed/creditcard_test.csv
  - models/creditcard_scaler.pkl


## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two separate pipelines | Fraud_Data has rich behavioral features needing engineering; creditcard is already PCA-transformed — the steps are fundamentally different |
| SMOTETomek as default | Fraud feature distributions overlap with legitimate transactions; Tomek links clean ambiguous boundary samples that plain SMOTE would amplify as noise |
| Fit transformers on train only | Fitting on full data leaks test set statistics (mean, std, categories) into training, artificially inflating evaluation metrics |
| StandardScaler over MinMaxScaler | Features have very different scales and may contain outliers; StandardScaler is more robust than MinMaxScaler in the presence of outliers |
| drop='first' in OneHotEncoder | Prevents the dummy variable trap (perfect multicollinearity) which destabilizes linear models like Logistic Regression |
| AUC-PR and F1 over Accuracy | With <10% fraud, a model predicting all-legitimate achieves >90% accuracy while catching zero fraud — accuracy is a meaningless metric here |
| Binary search for IP lookup | IpAddress_to_Country uses ranges not exact keys — binary search on sorted bounds is O(log n) and handles ~138K ranges efficiently |


## Data Sources

| Dataset | Source |
|---|---|
| `Fraud_Data.csv` | Provided by 10 Academy / Adey Innovations Inc. |
| `IpAddress_to_Country.csv` | Provided by 10 Academy / Adey Innovations Inc. |
| `creditcard.csv` | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

> The creditcard dataset was collected and analysed during a research
> collaboration by Worldline and the Machine Learning Group of ULB
> (Université Libre de Bruxelles).


## Dependencies

Key libraries used:

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, splitting, baseline models |
| `imbalanced-learn` | SMOTE, SMOTETomek, undersampling |
| `xgboost` / `lightgbm` | Ensemble models (Task 2) |
| `shap` | Model explainability (Task 3) |
| `joblib` | Saving fitted transformers |

Full dependency list in `requirements.txt`.


## Contributing

This project follows a branch-per-task workflow:

| Branch | Purpose |
|---|---|
| `main` | Stable, reviewed code only |
| `task-1` | Data analysis and preprocessing |
| `task-2` | Model building and training |
| `task-3` | Model explainability |

All work is done on task branches and merged to main via pull request.

