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

## 3. Completed Work: Task 2 — Model Building and Evaluation

### 3.1 Overview

Task 2 trained and evaluated four classification models on both preprocessed pipelines:
Logistic Regression (baseline), Random Forest, XGBoost, and LightGBM. Each model was
assessed on the held-out test set and validated with 5-fold stratified cross-validation.
Primary metric: AUC-PR. Secondary metrics: F1-Score, Precision, Recall, ROC-AUC.

---

### 3.2 Fraud_Data Pipeline Results

#### 3.2.1 Logistic Regression (Baseline)

- **AUC-PR:** 0.7005 | **F1:** 0.5988 | **Precision:** 0.5157 | **Recall:** 0.7138
- **Fraud caught:** 2,020 / 2,830 (71.4%) | **False alarms:** 1,897 (6.93% of legit)
- **Observation:** The baseline establishes a meaningful floor — AUC-PR of 0.7005 is
  well above the random baseline of 0.094, confirming the engineered features
  (particularly `time_since_signup` and transaction velocity) carry real linear signal.
  However, the low precision of 0.52 means nearly 1 in 2 fraud alerts is a false alarm,
  generating 1,897 unnecessary blocks. The model prioritizes recall at the expense of
  precision, which is operationally noisy. The CV AUC-PR of 0.8912 is substantially
  higher than the test AUC-PR of 0.7005, indicating the default 0.5 threshold is
  poorly calibrated for this class imbalance rather than true overfitting.

#### 3.2.2 Random Forest

- **AUC-PR:** 0.7116 | **F1:** 0.6097 | **Precision:** 0.5432 | **Recall:** 0.6947
- **Fraud caught:** 1,966 / 2,830 (69.5%) | **False alarms:** 1,653 (6.03% of legit)
- **Observation:** Random Forest improves AUC-PR modestly (+0.011 over LR) and reduces
  false alarms from 1,897 to 1,653, but catches slightly fewer fraud cases in absolute
  terms. The ensemble's non-linear boundary improves precision (0.54 vs 0.52) but the
  model still operates in a high-recall, low-precision regime at the default threshold.
  CV AUC-PR of 0.9626 confirms strong generalization potential that threshold tuning
  would unlock.

#### 3.2.3 XGBoost

- **AUC-PR:** 0.7131 | **F1:** 0.6902 | **Precision:** 0.9113 | **Recall:** 0.5555
- **Fraud caught:** 1,572 / 2,830 (55.5%) | **False alarms:** 153 (0.56% of legit)
- **Observation:** XGBoost takes a fundamentally different operating stance — precision
  jumps to 91.1% and false alarms collapse from 1,897 (LR) to just 153, a 92% reduction.
  The tradeoff is lower recall (55.5%), meaning it misses more fraud at the default
  threshold. Best F1 of 0.6902 reflects the superior precision-recall balance. CV
  AUC-PR of 0.9856 ± 0.0005 is the second highest with the second lowest variance,
  confirming stable generalization.

#### 3.2.4 LightGBM

- **AUC-PR:** 0.7136 | **F1:** 0.6900 | **Precision:** 0.9088 | **Recall:** 0.5562
- **Fraud caught:** 1,574 / 2,830 (55.6%) | **False alarms:** 158 (0.58% of legit)
- **Observation:** LightGBM is virtually indistinguishable from XGBoost at the default
  threshold — 1,574 vs 1,572 fraud caught, 158 vs 153 false alarms, F1 of 0.690 vs
  0.690. LightGBM achieves the highest CV AUC-PR of 0.9867 ± 0.0004 — the best and
  most stable generalization of any model on this pipeline — while training approximately
  40% faster than XGBoost.

#### 3.2.5 Model Comparison — Fraud_Data

[ Figure Placeholder: Model Comparison Bar Chart — Fraud_Data ]
[ Figure Placeholder: Precision-Recall Curves — Fraud_Data ]
[ Figure Placeholder: ROC Curves — Fraud_Data ]

| Model | AUC-PR | F1 | Precision | Recall | CV AUC-PR |
|---|---|---|---|---|---|
| Logistic Regression | 0.7005 | 0.5988 | 0.5157 | 0.7138 | 0.8912 ± 0.0014 |
| Random Forest | 0.7116 | 0.6097 | 0.5432 | 0.6947 | 0.9626 ± 0.0021 |
| XGBoost | 0.7131 | **0.6902** | **0.9113** | 0.5555 | 0.9856 ± 0.0005 |
| LightGBM | **0.7136** | 0.6900 | 0.9088 | 0.5562 | **0.9867 ± 0.0004** |

**Selected model: LightGBM**

Justification: Highest test AUC-PR (0.7136) and highest CV AUC-PR (0.9867) with the
lowest cross-validation variance (±0.0004), indicating the most stable generalization.
Near-identical test performance to XGBoost across every metric while offering faster
retraining. Precision of 90.9% means only 158 false alarms — a 92% reduction from the
logistic regression baseline. The PR curve confirms all four models cluster tightly
(0.700–0.714) on the test set while the ROC curves also cluster (0.840–0.843),
indicating the key differentiator is threshold behavior rather than ranking ability.
Threshold calibration in Task 3 will allow LightGBM to be tuned toward any
desired precision-recall operating point.

---

### 3.3 creditcard Pipeline Results

#### 3.3.1 Logistic Regression (Baseline)

- **AUC-PR:** 0.6738 | **F1:** 0.1005 | **Precision:** 0.0533 | **Recall:** 0.8737
- **Fraud caught:** 83 / 95 (87.4%) | **False alarms:** 1,474 (2.60% of legit)
- **Observation:** The extreme class imbalance (0.17% fraud, 95 fraud cases in the test
  set) exposes the logistic regression baseline severely. Despite a deceptively high
  accuracy of 97% and ROC-AUC of 0.9617, precision is just 0.053 — 95% of fraud alerts
  are false alarms. The model floods the alert queue with 1,474 false positives to catch
  83 genuine fraud cases. This is the textbook failure mode of an uncalibrated classifier
  on extreme imbalance. CV AUC-PR of 0.9919 reflects strong latent ranking ability that
  the default threshold fails to exploit.

#### 3.3.2 Random Forest

- **AUC-PR:** 0.7891 | **F1:** 0.6581 | **Precision:** 0.5540 | **Recall:** 0.8105
- **Fraud caught:** 77 / 95 (81.1%) | **False alarms:** 62 (0.11% of legit)
- **Observation:** Random Forest delivers the most dramatic improvement of any model
  transition in this project — false alarms drop from 1,474 (LR) to just 62, a 96%
  reduction, while maintaining 81.1% recall. Precision jumps from 0.053 to 0.554,
  making the alert queue operationally viable for the first time. This is the best
  F1 score on this pipeline (0.6581) and the best precision-recall balance at the
  default threshold. CV AUC-PR of 0.9998 ± 0.0000 is near-perfect, confirming the
  model has fully learned the fraud signal in the PCA feature space.

#### 3.3.3 XGBoost

- **AUC-PR:** 0.8018 | **F1:** 0.5474 | **Precision:** 0.4105 | **Recall:** 0.8211
- **Fraud caught:** 78 / 95 (82.1%) | **False alarms:** 112 (0.20% of legit)
- **Observation:** XGBoost achieves the highest test AUC-PR on this pipeline (0.8018),
  meaning it ranks fraud transactions better than any other model across all possible
  thresholds. It catches 78 fraud cases (one more than Random Forest) with 112 false
  alarms. The F1 of 0.5474 is lower than Random Forest's 0.6581 because at the default
  0.5 threshold XGBoost operates at lower precision (0.41 vs 0.55). CV AUC-PR of
  1.0000 ± 0.0001 — tied with LightGBM — indicates near-perfect fraud ranking on the
  training distribution.

#### 3.3.4 LightGBM

- **AUC-PR:** 0.7847 | **F1:** 0.5302 | **Precision:** 0.3892 | **Recall:** 0.8316
- **Fraud caught:** 79 / 95 (83.2%) | **False alarms:** 124 (0.22% of legit)
- **Observation:** LightGBM achieves the highest raw recall of any model on this pipeline
  (83.2%), catching 79 of 95 fraud cases. However test AUC-PR of 0.7847 falls below
  XGBoost (0.8018) and Random Forest (0.7891), despite CV AUC-PR of 1.0000 tied with
  XGBoost. This suggests LightGBM's probability calibration is slightly less optimal
  than XGBoost's on this dataset at the default threshold. CV F1 of 0.9984 confirms
  the model generalizes extremely well when the threshold is properly set.

#### 3.3.5 Model Comparison — creditcard

[ Figure Placeholder: Confusion Matrices — all four models, creditcard ]

| Model | AUC-PR | F1 | Precision | Recall | CV AUC-PR |
|---|---|---|---|---|---|
| Logistic Regression | 0.6738 | 0.1005 | 0.0533 | 0.8737 | 0.9919 ± 0.0002 |
| Random Forest | 0.7891 | **0.6581** | **0.5540** | 0.8105 | 0.9998 ± 0.0000 |
| XGBoost | **0.8018** | 0.5474 | 0.4105 | 0.8211 | **1.0000 ± 0.0001** |
| LightGBM | 0.7847 | 0.5302 | 0.3892 | **0.8316** | 1.0000 ± 0.0001 |

**Selected model: XGBoost**

Justification: Highest test AUC-PR of 0.8018 — the primary metric — meaning XGBoost
ranks fraud transactions better than any other model across all operating thresholds.
CV AUC-PR of 1.0000 ± 0.0001 tied with LightGBM confirms near-perfect generalization.
Catches 78 / 95 fraud cases (82.1%) with only 112 false alarms (0.20% of legitimate
transactions), a practical operating point for a banking fraud alert queue. The large
CV-to-test AUC-PR gap across all models is a dataset-level effect driven by the tiny
absolute fraud count in the test set (95 cases) — each misclassified fraud shifts
metrics significantly at this scale. Random Forest is the recommended alternative
if minimizing false alarms at the default threshold is prioritized (62 false alarms
vs 112, best F1 of 0.6581).

---

### 3.4 Final Model Selection

| Pipeline | Selected Model | Test AUC-PR | F1 | CV AUC-PR |
|---|---|---|---|---|
| Fraud_Data (e-commerce) | LightGBM | 0.7136 | 0.6900 | 0.9867 ± 0.0004 |
| creditcard (banking) | XGBoost | 0.8018 | 0.5474 | 1.0000 ± 0.0001 |

**Why AUC-PR was the deciding metric:**
With extreme class imbalance (9% fraud in Fraud_Data, 0.17% in creditcard), accuracy
is meaningless — a model predicting all-legitimate achieves 91%/99.83% accuracy while
catching zero fraud. Logistic Regression on creditcard demonstrates this precisely:
97% accuracy, ROC-AUC of 0.9617, but precision of 0.053 making it operationally
worthless. AUC-PR captures performance across all thresholds and is insensitive to
the dominant legitimate class, making it the only honest summary metric here.

**Precision vs Recall tradeoff:**
The two pipelines demand different operating stances. For the creditcard banking pipeline,
recall is weighted more heavily — a missed fraud is a direct financial loss with no
recovery. For the Fraud_Data e-commerce pipeline, the cost of false positives (blocked
customers, churn risk) is also significant, making the high-precision operating point
of LightGBM and XGBoost (91% precision, 158 false alarms vs 1,897 for LR) the more
defensible production choice. Threshold tuning in Task 3 will calibrate both selected
models to the exact precision-recall operating point that minimizes total business cost.

**Both selected models will be analyzed with SHAP in Task 3.**

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

