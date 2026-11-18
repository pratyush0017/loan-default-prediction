"""
train.py
--------
Model training, cross-model comparison, feature importance extraction,
and final model training on full data for Loan Default Prediction.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool

from src.preprocessing import load_and_preprocess_data, clean_dataframe, LoanPreprocessor
from src.evaluate import evaluate_models


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: str = "outputs/plots", top_n: int = 20):
    """
    Plots and saves feature importance horizontal bar chart.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_df = importance_df.head(top_n).sort_values(by="Importance", ascending=True)

    plt.figure(figsize=(10, 8))
    bars = plt.barh(top_df["Feature"], top_df["Importance"], color="#3498db", edgecolor="black")
    plt.title(f"Top {top_n} Predictive Features (CatBoost Importance)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Feature Importance Score (PredictionValuesChange / Gain)", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2, f"{width:.2f}",
                 va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def train_and_compare_models(train_path: str = "data/loan_train.csv", models_dir: str = "models", outputs_dir: str = "outputs"):
    """
    Trains Logistic Regression, Random Forest, and CatBoost models,
    evaluates them, extracts feature importance, and trains the final model on full dataset.
    """
    os.makedirs(models_dir, exist_ok=True)
    plots_dir = os.path.join(outputs_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 65)
    print("STEP 1: PREPARING DATA & PREPROCESSING PIPELINE")
    print("=" * 65)
    data = load_and_preprocess_data(train_path, test_size=0.2, random_state=42)

    X_train_sk, X_val_sk = data["X_train_sk"], data["X_val_sk"]
    X_train_cb, X_val_cb = data["X_train_cb"], data["X_val_cb"]
    y_train, y_val = data["y_train"], data["y_val"]
    cat_cols = data["categorical_cols"]

    print(f"Training Samples: {len(X_train_cb):,}, Validation Samples: {len(X_val_cb):,}")
    print(f"Categorical Features: {cat_cols}")

    # ==========================================
    # 1. Logistic Regression
    # ==========================================
    print("\n" + "=" * 65)
    print("STEP 2: TRAINING MODEL 1 - LOGISTIC REGRESSION (BALANCED)")
    print("=" * 65)
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    lr_model.fit(X_train_sk, y_train)
    lr_pred = lr_model.predict(X_val_sk)
    lr_prob = lr_model.predict_proba(X_val_sk)[:, 1]

    # ==========================================
    # 2. Random Forest Classifier
    # ==========================================
    print("\n" + "=" * 65)
    print("STEP 3: TRAINING MODEL 2 - RANDOM FOREST CLASSIFIER")
    print("=" * 65)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_sk, y_train)
    rf_pred = rf_model.predict(X_val_sk)
    rf_prob = rf_model.predict_proba(X_val_sk)[:, 1]

    # ==========================================
    # 3. CatBoost Classifier
    # ==========================================
    print("\n" + "=" * 65)
    print("STEP 4: TRAINING MODEL 3 - CATBOOST CLASSIFIER")
    print("=" * 65)
    # Calculate scale_pos_weight based on class ratio
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    cat_indices = [X_train_cb.columns.get_loc(c) for c in cat_cols]
    train_pool = Pool(X_train_cb, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val_cb, y_val, cat_features=cat_indices)

    cb_model = CatBoostClassifier(
        iterations=700,
        depth=7,
        learning_rate=0.05,
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )
    cb_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    cb_prob = cb_model.predict_proba(X_val_cb)[:, 1]
    # Optimal probability threshold for balanced classes or default 0.5
    cb_pred = (cb_prob >= 0.5).astype(int)

    # ==========================================
    # 4. Model Comparison & Evaluation
    # ==========================================
    models_dict = {
        "Logistic Regression": {"y_pred": lr_pred, "y_prob": lr_prob},
        "Random Forest": {"y_pred": rf_pred, "y_prob": rf_prob},
        "CatBoost": {"y_pred": cb_pred, "y_prob": cb_prob}
    }

    comparison_df = evaluate_models(models_dict, y_val, output_dir=plots_dir)
    print("\nModel Comparison Summary Table:")
    print(comparison_df.to_string())

    # ==========================================
    # 5. Feature Importance from CatBoost
    # ==========================================
    print("\n" + "=" * 65)
    print("STEP 5: EXTRACTING & SAVING FEATURE IMPORTANCE")
    print("=" * 65)
    importances = cb_model.get_feature_importance()
    feat_names = list(X_train_cb.columns)
    importance_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    csv_fi_path = os.path.join(outputs_dir, "feature_importance.csv")
    importance_df.to_csv(csv_fi_path, index=False)
    print(f"Top 10 Important Features:\n{importance_df.head(10).to_string()}")
    print(f"Feature importance saved to: {csv_fi_path}")

    plot_feature_importance(importance_df, output_dir=plots_dir, top_n=20)

    # ==========================================
    # 6. Final Model: Retrain on Full Dataset
    # ==========================================
    print("\n" + "=" * 65)
    print("STEP 6: TRAINING FINAL CATBOOST MODEL ON FULL DATASET")
    print("=" * 65)
    raw_full = pd.read_csv(train_path)
    clean_full = clean_dataframe(raw_full, is_train=True)
    X_full_raw = clean_full.drop(columns=["loan_status"])
    y_full = clean_full["loan_status"].astype(int)

    full_preprocessor = LoanPreprocessor()
    full_preprocessor.fit(X_full_raw)
    X_full_cb = full_preprocessor.transform_catboost(X_full_raw)

    best_iter = cb_model.get_best_iteration()
    final_iterations = best_iter + 50 if best_iter is not None and best_iter > 0 else 700
    print(f"Retraining final model on {len(X_full_cb):,} samples with {final_iterations} iterations...")

    full_pool = Pool(X_full_cb, y_full, cat_features=cat_indices)
    final_catboost = CatBoostClassifier(
        iterations=final_iterations,
        depth=7,
        learning_rate=0.05,
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100
    )
    final_catboost.fit(full_pool)

    # Save final model bundle
    final_model_bundle = {
        "model": final_catboost,
        "preprocessor": full_preprocessor,
        "categorical_cols": cat_cols,
        "feature_names": full_preprocessor.feature_names,
        "model_name": "CatBoostClassifier",
        "threshold": 0.5
    }

    bundle_path = os.path.join(models_dir, "final_model.joblib")
    joblib.dump(final_model_bundle, bundle_path)
    # Also save native catboost model format
    cbm_path = os.path.join(models_dir, "catboost_model.cbm")
    final_catboost.save_model(cbm_path)
    print(f"\n[OK] Final model bundle saved successfully to: {bundle_path}")
    print(f"[OK] Native CatBoost model saved to: {cbm_path}")

    return {
        "comparison_df": comparison_df,
        "final_model_bundle": final_model_bundle,
        "importance_df": importance_df
    }


if __name__ == "__main__":
    train_and_compare_models()

