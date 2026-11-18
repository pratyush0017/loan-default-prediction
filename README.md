# 🏦 End-to-End Loan Default Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/ML%20Framework-CatBoost%20%7C%20Scikit--Learn-orange.svg)](https://catboost.ai/)
[![UI](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, production-grade Machine Learning solution built exclusively in **Python** to assess credit risk and predict whether a loan applicant is likely to default (`loan_status = 1`) or fully repay (`loan_status = 0`).

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
3. [Dataset Description](#-dataset-description)
4. [Data Cleaning & Anti-Leakage Controls](#-data-cleaning--anti-leakage-controls)
5. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
6. [Feature Engineering & Preprocessing Pipeline](#-feature-engineering--preprocessing-pipeline)
7. [Model Training & Evaluation Benchmark](#-model-training--evaluation-benchmark)
8. [Feature Importance Insights](#-feature-importance-insights)
9. [Streamlit Interactive Web Application](#-streamlit-interactive-web-application)
10. [Installation & Setup](#-installation--setup)
11. [How to Run Training & Inference](#-how-to-run-training--inference)

---

## 📌 Project Overview & Problem Statement

Credit default risk is one of the most critical challenges facing commercial banks, peer-to-peer lending platforms, and fintech lenders. Approving loans for borrowers who subsequently default leads to substantial non-performing assets (NPAs) and capital impairment.

### Objective
Build a binary classification machine learning system that:
- Ingests borrower demographic, credit history, and loan characteristics.
- Preprocesses and cleans raw data without target leakage.
- Benchmarks multiple algorithms (**Logistic Regression**, **Random Forest**, and **CatBoost**).
- Delivers an intuitive **Streamlit** dashboard and real-time inference tool.

**Technology Constraint**: Built **100% in Python** with zero JavaScript / Node.js / React dependencies.

---

## 🗂️ Project Architecture & Directory Structure

```text
loan-default-prediction/
├── data/
│   ├── loan_train.csv                   # Historical training dataset (27,003 records)
│   ├── loan_test.csv                    # Evaluation test dataset (11,574 records)
│   └── loan_sample_submission.csv       # Sample submission format template
├── notebooks/
│   └── loan_default_prediction.ipynb    # End-to-end tutorial and analysis notebook
├── src/
│   ├── preprocessing.py                 # Cleaning, parsing, and pipeline transformations
│   ├── eda.py                           # EDA summary statistics and plot generation
│   ├── train.py                         # Model training, comparison, and persistence
│   ├── evaluate.py                      # Metrics, confusion matrix, ROC/PR curves
│   └── predict.py                       # Test inference & batch submission generator
├── models/
│   ├── final_model.joblib               # Persisted champion model bundle + preprocessor
│   └── catboost_model.cbm               # Native CatBoost model artifact
├── outputs/
│   ├── feature_importance.csv           # Ranked feature importance scores
│   ├── loan_submission.csv              # Test set predictions (id, loan_status)
│   └── plots/                           # High-resolution visualization figures
│       ├── target_distribution.png
│       ├── numerical_distributions.png
│       ├── categorical_distributions.png
│       ├── correlation_heatmap.png
│       ├── int_rate_by_status.png
│       ├── grade_vs_default.png
│       ├── dti_vs_annual_inc.png
│       ├── roc_curves.png
│       ├── pr_curves.png
│       ├── confusion_matrices.png
│       ├── model_comparison.png
│       └── feature_importance.png
├── app.py                               # Interactive Streamlit Web Application
├── requirements.txt                     # Pure Python dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 📊 Dataset Description

The project uses lending club loan records comprising 47 raw attributes.

| Dataset Split | Record Count | Feature Count | Target Present |
| :--- | :--- | :--- | :--- |
| **Train Set** (`loan_train.csv`) | 27,003 | 47 | Yes (`loan_status`) |
| **Test Set** (`loan_test.csv`) | 11,574 | 46 | No (to be predicted) |

### Target Distribution (Class Imbalance)
- **Non-Default (`0`)**: 23,026 records (**85.27%**)
- **Default (`1`)**: 3,977 records (**14.73%**)
- **Imbalance Ratio**: ~ 5.8 : 1

---

## 🛡️ Data Cleaning & Anti-Leakage Controls

### 1. Data Leakage Removal
Post-loan tracking features are recorded *after* a borrower begins repayment or defaults. Using these during credit underwriting would falsely inflate model accuracy:
```python
LEAKAGE_COLUMNS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "last_credit_pull_d"
]
```

### 2. Identifier & Free-Text Drops
Non-predictive IDs and raw URLs/descriptions were removed: `['id', 'member_id', 'url', 'desc']`. Free-text high cardinality fields (`emp_title`, `title`, `zip_code`) were also dropped to prevent overfitting.

### 3. Data Type Sanitation
- `int_rate`: Converted from string (`"13.23%"`) $\rightarrow$ numeric (`13.23`).
- `revol_util`: Converted from string (`"62.70%"`) $\rightarrow$ numeric (`62.70`).
- `term`: Extracted numeric months (`" 36 months"` $\rightarrow$ `36`, `" 60 months"` $\rightarrow$ `60`).
- `emp_length`: Standardized `< 1 year` $\rightarrow$ `0.0`, `10+ years` $\rightarrow$ `10.0`, `n years` $\rightarrow$ `n.0`.

### 4. Date Feature Extraction
- `issue_d` $\rightarrow$ Extracted `issue_year` and `issue_month`.
- `earliest_cr_line` $\rightarrow$ Extracted `credit_history_year` and computed `credit_history_length_years` (`issue_year - earliest_cr_year`).

---

## 📈 Exploratory Data Analysis (EDA)

Key statistical findings from the dataset:
1. **Higher Interest Rates for Defaults**: The median interest rate for defaulting borrowers is **13.5%**, compared to **11.2%** for non-defaulters.
2. **Grade Monotonicity**: Credit Grade A has only a **6.0%** default rate, whereas Grade G exhibits a **33.7%** default rate.
3. **Debt-to-Income (DTI)**: Borrowers with DTI > 20% show significantly elevated default propensity.

All EDA charts are generated and saved under `outputs/plots/`.

---

## ⚙️ Feature Engineering & Preprocessing Pipeline

To guarantee zero data leakage between splits:
- **Imputation**:
  - Continuous missing values are imputed using the **training set median**.
  - Categorical missing values are mapped to `"Unknown"`.
- **Encodings**:
  - **CatBoost**: Utilizes native categorical handling directly on categorized string features (`grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`).
  - **Scikit-Learn (Logistic Regression & Random Forest)**: Utilizes `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore')` for categoricals.

---

## 🏆 Model Training & Evaluation Benchmark

We performed a stratified 80/20 split (`21,602` training / `5,401` validation) and evaluated using imbalanced classification metrics:

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 66.45% | 0.2481 | 63.02% | 0.3561 | 0.7138 |
| **Random Forest (Balanced)** | 69.36% | 0.2582 | 57.74% | 0.3568 | 0.7065 |
| **CatBoost Classifier (Champion)** | **69.56%** | **0.2651** | **60.25%** | **0.3682** | **0.7227** |

> **Champion Selection**: **CatBoost Classifier** achieved the highest ROC-AUC (**0.7227**), highest F1 Score (**0.3682**), and highest overall accuracy (**69.56%**). It was retrained on the complete 100% training dataset (`data/loan_train.csv`) and saved to `models/final_model.joblib`.

---

## 🔍 Feature Importance Insights

Top predictive features extracted from CatBoost feature importance:

| Rank | Feature Name | Description | Importance Gain |
| :---: | :--- | :--- | :---: |
| **1** | `annual_inc` | Borrower annual stated income | **11.87%** |
| **2** | `purpose` | Loan purpose category (e.g. debt consolidation) | **6.82%** |
| **3** | `int_rate` | Assigned interest rate percentage | **6.64%** |
| **4** | `addr_state` | US State of borrower residence | **6.62%** |
| **5** | `grade` | Assigned credit risk grade (A - G) | **6.47%** |
| **6** | `revol_util` | Credit line revolving utilization percentage | **5.37%** |
| **7** | `revol_bal` | Total revolving credit balance ($) | **4.67%** |
| **8** | `sub_grade` | Granular credit rating | **4.46%** |
| **9** | `issue_year` | Year the loan was issued (macroeconomic vintage) | **4.21%** |
| **10** | `term` | Loan tenure (36 vs 60 months) | **3.88%** |

---

## 🖥️ Streamlit Interactive Web Application

The interactive web UI (`app.py`) provides:
- **🎯 Real-Time Loan Risk Predictor**: Form to enter loan amount, interest rate, income, credit grade, DTI, etc., and obtain instant risk classification, default probability %, and approval recommendations.
- **📊 Exploratory Data Dashboard**: Interactive KPIs, distributions, and relationship charts.
- **🏆 Model Performance Benchmarks**: ROC curves, PR curves, Confusion matrices, and feature importance rankings.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.14 Anaconda environment)

### 1. Clone or Open the Repository
```bash
git clone <repo_url>
cd loan-default-prediction
```

### 2. Install Required Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 How to Run Training & Inference

### 1. Run Exploratory Data Analysis (EDA)
Generates summary statistics and exports high-resolution plots to `outputs/plots/`:
```bash
python src/eda.py
```

### 2. Train and Benchmark Models
Trains Logistic Regression, Random Forest, and CatBoost, evaluates them, extracts feature importance, and trains the final production model:
```bash
python src/train.py
```

### 3. Generate Test Predictions (`outputs/loan_submission.csv`)
Generates predictions on `data/loan_test.csv`:
```bash
python src/predict.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📄 Final Submission Verification

The batch prediction file `outputs/loan_submission.csv` is verified:
- **Exact Shape**: `(11574, 2)`
- **Header**: `id,loan_status`
- **Zero Nulls**: 100% filled predictions aligned with `loan_test.csv` IDs.

