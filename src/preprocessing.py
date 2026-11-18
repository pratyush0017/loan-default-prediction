"""
preprocessing.py
----------------
Data cleaning, feature engineering, and preprocessing pipeline for
Loan Default Prediction.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Post-loan and data leakage columns strictly removed before modeling
LEAKAGE_COLUMNS = [
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d"
]

# Non-predictive identifier and heavy text fields
DROP_COLUMNS = ["id", "member_id", "url", "desc"]

# High cardinality free-text columns that do not generalize well
HIGH_CARDINALITY_TEXT = ["emp_title", "title", "zip_code"]


def parse_date_series(series):
    """
    Parses 'Mon-YY' format strings (e.g., 'Jul-10', 'May-99')
    and handles century cutoff correctly.
    """
    dt = pd.to_datetime(series, format="%b-%y", errors="coerce")
    # If year > 2025, adjust to 1900s (e.g. 1999)
    dt = dt.map(lambda d: d.replace(year=d.year - 100) if pd.notna(d) and d.year > 2025 else d)
    return dt


def parse_emp_length(val):
    """
    Extracts numeric employment length in years.
    '< 1 year' -> 0
    '10+ years' -> 10
    '1 year' ... '9 years' -> 1 ... 9
    NaN / None -> np.nan
    """
    if pd.isna(val) or val == "" or str(val).lower() == "nan":
        return np.nan
    val_str = str(val).strip()
    if "<" in val_str:
        return 0.0
    if "10+" in val_str:
        return 10.0
    m = re.search(r"(\d+)", val_str)
    return float(m.group(1)) if m else np.nan


def clean_dataframe(df: pd.DataFrame, is_train: bool = True, drop_high_cardinality: bool = True) -> pd.DataFrame:
    """
    Applies standard cleaning and feature engineering to a DataFrame.
    """
    df = df.copy()

    # 1. Drop post-loan / data leakage columns
    leakage_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leakage_to_drop:
        df = df.drop(columns=leakage_to_drop)

    # 2. Drop identifiers / non-predictive columns
    # Keep 'id' only if is_train is False and caller might need it, but clean_dataframe drops it if present
    drops = [c for c in DROP_COLUMNS if c in df.columns]
    if drops:
        df = df.drop(columns=drops)

    # Optional: drop very high cardinality text features
    if drop_high_cardinality:
        text_drops = [c for c in HIGH_CARDINALITY_TEXT if c in df.columns]
        if text_drops:
            df = df.drop(columns=text_drops)

    # 3. Clean int_rate -> numeric
    if "int_rate" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["int_rate"]):
            df["int_rate"] = df["int_rate"].astype(str).str.rstrip("%").str.strip()
            df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    # 4. Clean revol_util -> numeric
    if "revol_util" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["revol_util"]):
            df["revol_util"] = df["revol_util"].astype(str).str.rstrip("%").str.strip()
            df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

    # 5. Clean term -> numeric
    if "term" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["term"]):
            df["term"] = df["term"].astype(str).str.extract(r"(\d+)")[0]
            df["term"] = pd.to_numeric(df["term"], errors="coerce")

    # 6. Clean emp_length -> numeric
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].apply(parse_emp_length)

    # 7. Date parsing & feature creation
    if "issue_d" in df.columns:
        issue_dt = parse_date_series(df["issue_d"])
        df["issue_year"] = issue_dt.dt.year
        df["issue_month"] = issue_dt.dt.month
        df = df.drop(columns=["issue_d"])
    else:
        if "issue_year" not in df.columns:
            df["issue_year"] = np.nan
        if "issue_month" not in df.columns:
            df["issue_month"] = np.nan

    if "earliest_cr_line" in df.columns:
        cr_dt = parse_date_series(df["earliest_cr_line"])
        earliest_cr_year = cr_dt.dt.year
        df["credit_history_year"] = earliest_cr_year
        if "issue_year" in df.columns:
            df["credit_history_length_years"] = (df["issue_year"] - earliest_cr_year).clip(lower=0)
        df = df.drop(columns=["earliest_cr_line"])

    return df


class LoanPreprocessor:
    """
    Encapsulates training state (numeric medians, categorical mappings,
    sklearn ColumnTransformer) for consistent train, validation, and test processing.
    """

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.numeric_medians = {}
        self.sklearn_pipeline = None
        self.feature_names = []

    def fit(self, X: pd.DataFrame):
        """
        Learns column types and statistics from training features.
        """
        # Automatically identify categorical and numeric columns
        self.numeric_cols = []
        self.categorical_cols = []
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                self.numeric_cols.append(c)
            else:
                self.categorical_cols.append(c)

        # Calculate medians for numeric columns
        for col in self.numeric_cols:
            num_series = pd.to_numeric(X[col], errors="coerce")
            med = num_series.median()
            self.numeric_medians[col] = 0.0 if pd.isna(med) else float(med)

        self.feature_names = list(X.columns)

        # Build Sklearn preprocessing pipeline for Logistic Regression / Random Forest
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.sklearn_pipeline = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols)
            ]
        )
        self.sklearn_pipeline.fit(X)
        return self

    def transform_catboost(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms DataFrame for CatBoost:
        - Numeric missing filled with learned medians
        - Categorical missing filled with 'Unknown' and cast to string
        """
        X_out = X.copy()
        
        # Ensure all columns present
        for col in self.feature_names:
            if col not in X_out.columns:
                if col in self.numeric_cols:
                    X_out[col] = self.numeric_medians[col]
                else:
                    X_out[col] = "Unknown"

        # Reorder to match training features
        X_out = X_out[self.feature_names]

        # Fill missing values
        for col in self.numeric_cols:
            X_out[col] = X_out[col].fillna(self.numeric_medians.get(col, 0.0))
            X_out[col] = pd.to_numeric(X_out[col], errors="coerce").fillna(self.numeric_medians.get(col, 0.0))

        for col in self.categorical_cols:
            X_out[col] = X_out[col].fillna("Unknown").astype(str)
            # Replace empty strings or string 'nan'
            X_out[col] = X_out[col].replace({"nan": "Unknown", "": "Unknown"})

        return X_out

    def transform_sklearn(self, X: pd.DataFrame):
        """
        Transforms DataFrame for Sklearn models (One-Hot Encoded + Scaled Numeric).
        """
        X_cb = self.transform_catboost(X)
        return self.sklearn_pipeline.transform(X_cb)


def load_and_preprocess_data(train_csv_path: str = "data/loan_train.csv", test_size: float = 0.2, random_state: int = 42):
    """
    Loads raw training CSV, applies cleaning, fits preprocessor, and splits into train/validation sets.
    """
    raw_df = pd.read_csv(train_csv_path)
    clean_df = clean_dataframe(raw_df, is_train=True)

    target_col = "loan_status"
    if target_col not in clean_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col].astype(int)

    # Train / Validation Stratified Split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = LoanPreprocessor()
    preprocessor.fit(X_train_raw)

    X_train_cb = preprocessor.transform_catboost(X_train_raw)
    X_val_cb = preprocessor.transform_catboost(X_val_raw)

    X_train_sk = preprocessor.transform_sklearn(X_train_raw)
    X_val_sk = preprocessor.transform_sklearn(X_val_raw)

    return {
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_train_cb": X_train_cb,
        "X_val_cb": X_val_cb,
        "X_train_sk": X_train_sk,
        "X_val_sk": X_val_sk,
        "y_train": y_train,
        "y_val": y_val,
        "categorical_cols": preprocessor.categorical_cols,
        "numeric_cols": preprocessor.numeric_cols,
        "feature_names": preprocessor.feature_names,
        "preprocessor": preprocessor
    }
