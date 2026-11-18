"""
evaluate.py
-----------
Comprehensive model evaluation metrics and comparison visualization module.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, and generates ROC, PR,
and Confusion Matrix plots.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Computes all standard classification metrics.
    """
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob))
    }


def evaluate_models(models_dict: dict, y_val, output_dir: str = "outputs/plots") -> pd.DataFrame:
    """
    Evaluates multiple models, prints classification reports, computes metrics DataFrame,
    and saves Confusion Matrices, ROC Curves, and Precision-Recall Curves.

    Parameters:
    -----------
    models_dict: dict of {
        'model_name': {
            'y_pred': array-like,
            'y_prob': array-like
        }
    }
    y_val: true validation targets
    output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print("\n" + "=" * 65)
    print("MODEL VALIDATION EVALUATION RESULTS")
    print("=" * 65)

    for name, data in models_dict.items():
        y_pred = data["y_pred"]
        y_prob = data["y_prob"]

        metrics = compute_metrics(y_val, y_pred, y_prob)
        metrics["Model"] = name
        results.append(metrics)

        print(f"\n>>> Model: {name}")
        print("-" * 40)
        for k, v in metrics.items():
            if k != "Model":
                print(f"  {k:12s}: {v:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=["Non-Default (0)", "Default (1)"], digits=4))

    results_df = pd.DataFrame(results).set_index("Model")

    # 1. Plot Combined Confusion Matrices
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(models_dict.items()):
        cm = confusion_matrix(y_val, data["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Non-Default", "Default"],
                    yticklabels=["Non-Default", "Default"])
        axes[idx].set_title(f"Confusion Matrix\n{name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.suptitle("Validation Confusion Matrix Comparison", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Plot ROC Curves Comparison
    plt.figure(figsize=(8, 6))
    for name, data in models_dict.items():
        fpr, tpr, _ = roc_curve(y_val, data["y_prob"])
        auc_val = roc_auc_score(y_val, data["y_prob"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curves Comparison Across Models", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Plot Precision-Recall Curves Comparison
    plt.figure(figsize=(8, 6))
    baseline_prec = float(np.mean(y_val))
    for name, data in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_val, data["y_prob"])
        f1_val = f1_score(y_val, data["y_pred"], zero_division=0)
        plt.plot(recall, precision, lw=2, label=f"{name} (F1 = {f1_val:.4f})")

    plt.axhline(y=baseline_prec, color="gray", linestyle="--", lw=1.5, label=f"No-Skill Baseline ({baseline_prec:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title("Precision-Recall Curves Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    metrics_to_plot = results_df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    metrics_to_plot.plot(kind="bar", figsize=(11, 5), colormap="viridis", edgecolor="black")
    plt.title("Model Performance Metrics Comparison on Validation Set", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=0, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[OK] Model evaluation completed. Evaluation plots generated in:", output_dir)
    return results_df

