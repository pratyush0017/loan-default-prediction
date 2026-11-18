"""
predict.py
----------
Inference and batch submission generation for Loan Default Prediction.
Loads the trained model artifact, applies identical preprocessing to test data,
and writes 'outputs/loan_submission.csv'.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import numpy as np
from src.preprocessing import clean_dataframe


def predict_test_dataset(test_csv_path: str = "data/loan_test.csv",
                         model_bundle_path: str = "models/final_model.joblib",
                         output_submission_path: str = "outputs/loan_submission.csv"):
    """
    Loads test dataset, processes it using stored preprocessor,
    generates binary predictions, and writes submission CSV.
    """
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at: {test_csv_path}")
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"Model bundle not found at: {model_bundle_path}. Please run train.py first.")

    print("=" * 65)
    print("STEP 1: LOADING TEST DATA & MODEL BUNDLE")
    print("=" * 65)
    test_raw = pd.read_csv(test_csv_path)
    print(f"Loaded test dataset: {test_raw.shape[0]:,} rows, {test_raw.shape[1]} columns")

    # Preserve test IDs for final submission
    if "id" not in test_raw.columns:
        raise ValueError("Column 'id' is required in test dataset for submission generation.")
    test_ids = test_raw["id"].copy()

    # Load model bundle
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = bundle.get("threshold", 0.5)

    print(f"Loaded model type: {type(model).__name__}")
    print(f"Classification threshold: {threshold}")

    print("\n" + "=" * 65)
    print("STEP 2: PREPROCESSING TEST DATA")
    print("=" * 65)
    test_clean = clean_dataframe(test_raw, is_train=False)
    test_transformed = preprocessor.transform_catboost(test_clean)
    print(f"Transformed test feature matrix: {test_transformed.shape}")

    print("\n" + "=" * 65)
    print("STEP 3: GENERATING PREDICTIONS")
    print("=" * 65)
    # Generate probabilities for Class 1 (Default)
    default_probabilities = model.predict_proba(test_transformed)[:, 1]
    predictions = (default_probabilities >= threshold).astype(int)

    pred_series = pd.Series(predictions)
    print("Prediction Class Distribution:")
    for val, count in pred_series.value_counts().items():
        label = "Non-Default (0)" if val == 0 else "Default (1)"
        print(f"  {label}: {count:,} ({count/len(predictions)*100:.2f}%)")

    print("\n" + "=" * 65)
    print("STEP 4: WRITING SUBMISSION CSV")
    print("=" * 65)
    submission_df = pd.DataFrame({
        "id": test_ids,
        "loan_status": predictions
    })

    os.makedirs(os.path.dirname(output_submission_path), exist_ok=True)
    submission_df.to_csv(output_submission_path, index=False)
    print(f"[OK] Submission saved to: {output_submission_path}")
    print(f"File size: {os.path.getsize(output_submission_path):,} bytes")
    print("\nFirst 10 rows of submission:")
    print(submission_df.head(10).to_string(index=False))

    return submission_df


def predict_single_loan(borrower_dict: dict, model_bundle_path: str = "models/final_model.joblib") -> dict:
    """
    Helper function for real-time inference in Streamlit app.
    Takes a dictionary of loan & borrower inputs and returns predictions + probabilities.
    """
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"Model bundle not found at: {model_bundle_path}")

    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    threshold = bundle.get("threshold", 0.5)

    input_df = pd.DataFrame([borrower_dict])
    cleaned_df = clean_dataframe(input_df, is_train=False)
    transformed_df = preprocessor.transform_catboost(cleaned_df)

    probs = model.predict_proba(transformed_df)[0]
    prob_non_default = float(probs[0])
    prob_default = float(probs[1])
    predicted_class = int(prob_default >= threshold)

    return {
        "predicted_class": predicted_class,
        "prediction_label": "High Risk / Default" if predicted_class == 1 else "Low Risk / Non-Default",
        "default_probability": prob_default,
        "non_default_probability": prob_non_default,
        "risk_level": "High" if prob_default >= 0.5 else ("Medium" if prob_default >= 0.25 else "Low")
    }


if __name__ == "__main__":
    predict_test_dataset()

