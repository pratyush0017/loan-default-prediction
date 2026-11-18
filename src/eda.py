"""
eda.py
------
Exploratory Data Analysis (EDA) module and script for Loan Default Prediction.
Generates comprehensive statistical summaries and publication-grade plots.
"""

import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import clean_dataframe

# Set modern plotting theme
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def perform_eda(data_path: str = "data/loan_train.csv", output_dir: str = "outputs/plots"):
    """
    Executes full EDA pipeline and writes visual summaries to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("LOAN DEFAULT PREDICTION - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)

    # 1. Load raw dataset
    df_raw = pd.read_csv(data_path)
    print(f"\n[1] Dataset Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Duplicate Check
    num_duplicates = df_raw.duplicated().sum()
    print(f"[2] Duplicate Rows Check: {num_duplicates} duplicate(s) found")

    # 3. Missing Values Summary
    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(by="Missing Count", ascending=False)
    print(f"\n[3] Missing Values in Raw Data ({len(missing_df)} columns with missing data):")
    print(missing_df.head(10).to_string())

    # 4. Target Distribution
    target_col = "loan_status"
    target_counts = df_raw[target_col].value_counts()
    target_pct = df_raw[target_col].value_counts(normalize=True) * 100
    print(f"\n[4] Target Distribution ('{target_col}'):")
    for val in target_counts.index:
        label = "Non-Default (0)" if val == 0 else "Default (1)"
        print(f"    {label}: {target_counts[val]:,} ({target_pct[val]:.2f}%)")

    # Save Plot 1: Target Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=target_col, hue=target_col, data=df_raw, palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Loan Status Distribution (0: Non-Default, 1: Default)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Loan Status", fontsize=11)
    plt.ylabel("Borrower Count", fontsize=11)
    plt.xticks([0, 1], ["Fully Paid / Non-Default (0)", "Charged Off / Default (1)"])
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:,.0f} ({height/len(df_raw)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=300)
    plt.close()

    # Cleaned DataFrame for Feature Analysis
    df_clean = clean_dataframe(df_raw, is_train=True)
    print(f"\n[5] Cleaned Feature Set Shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")

    # 5. Descriptive Statistics for Numeric Features
    numeric_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c]) and c != target_col]
    print("\n[6] Descriptive Statistics for Key Numeric Features:")
    desc_cols = [c for c in ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_util"] if c in numeric_cols]
    print(df_clean[desc_cols].describe().T[["mean", "std", "min", "50%", "max"]].to_string())

    # Save Plot 2: Numerical Feature Distributions (Histograms + KDE)
    key_num_features = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "revol_util"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, col in enumerate(key_num_features):
        if col in df_clean.columns:
            sns.histplot(df_clean[col].dropna(), kde=True, ax=axes[idx], color="#3498db", bins=30)
            axes[idx].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
    plt.suptitle("Distributions of Key Continuous Loan Features", fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "numerical_distributions.png"), dpi=300)
    plt.close()

    # Save Plot 3: Categorical Feature Distributions (Grade, Sub-Grade, Home Ownership, Purpose)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cat_to_plot = ["grade", "home_ownership", "verification_status", "purpose"]
    for idx, col in enumerate(cat_to_plot):
        ax = axes[idx // 2, idx % 2]
        if col in df_clean.columns:
            order = df_clean[col].value_counts().index[:10]
            sns.countplot(y=col, hue=target_col, data=df_clean, order=order,
                          palette=["#2ecc71", "#e74c3c"], ax=ax)
            ax.set_title(f"Distribution of {col} by Loan Status", fontsize=12, fontweight="bold")
            ax.set_xlabel("Count")
            ax.set_ylabel(col)
            ax.legend(["Non-Default (0)", "Default (1)"], loc="lower right")
    plt.suptitle("Categorical Distributions Stratified by Loan Status", fontsize=15, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(output_dir, "categorical_distributions.png"), dpi=300)
    plt.close()

    # Save Plot 4: Correlation Heatmap for Numeric Features
    plt.figure(figsize=(14, 11))
    corr_cols = [c for c in numeric_cols if c not in ["issue_year", "issue_month"]] + [target_col]
    corr_matrix = df_clean[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
                mask=mask, linewidths=0.5, annot_kws={"size": 8})
    plt.title("Correlation Heatmap of Numerical Features and Target", fontsize=14, fontweight="bold", pad=12)
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()

    # Save Plot 5: Interest Rate by Loan Status Boxplot & Violin Plot
    plt.figure(figsize=(9, 5))
    sns.boxplot(x=target_col, y="int_rate", hue=target_col, data=df_clean, palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Interest Rate Distribution by Loan Status (Defaults pay higher interest)", fontsize=13, fontweight="bold")
    plt.xlabel("Loan Status")
    plt.ylabel("Interest Rate (%)")
    plt.xticks([0, 1], ["Non-Default (0)", "Default (1)"])
    plt.savefig(os.path.join(output_dir, "int_rate_by_status.png"), dpi=300)
    plt.close()

    # Save Plot 6: Default Rate by Credit Grade
    if "grade" in df_clean.columns:
        plt.figure(figsize=(9, 5))
        grade_default = df_clean.groupby("grade")[target_col].mean().reset_index()
        grade_order = sorted(df_clean["grade"].dropna().unique())
        sns.barplot(x="grade", y=target_col, hue="grade", data=grade_default, order=grade_order, palette="Reds_r", legend=False)
        plt.title("Default Probability by Credit Grade (Grade A = Lowest Risk, G = Highest Risk)", fontsize=13, fontweight="bold")
        plt.xlabel("Credit Grade")
        plt.ylabel("Default Probability (Proportion)")
        for idx, row in grade_default.iterrows():
            plt.text(grade_order.index(row["grade"]), row[target_col] + 0.01, f"{row[target_col]*100:.1f}%",
                     ha="center", fontweight="bold")
        plt.ylim(0, max(grade_default[target_col]) * 1.25)
        plt.savefig(os.path.join(output_dir, "grade_vs_default.png"), dpi=300)
        plt.close()

    # Save Plot 7: DTI vs Annual Income Relationship
    if "dti" in df_clean.columns and "annual_inc" in df_clean.columns:
        plt.figure(figsize=(9, 5))
        # Filter income outliers for clearer scatter/kde view
        inc_filter = df_clean["annual_inc"] <= df_clean["annual_inc"].quantile(0.98)
        sns.scatterplot(x="annual_inc", y="dti", hue=target_col, data=df_clean[inc_filter],
                        palette=["#2ecc71", "#e74c3c"], alpha=0.4, s=30)
        plt.title("Debt-to-Income (DTI) vs Annual Income (<98th percentile)", fontsize=13, fontweight="bold")
        plt.xlabel("Annual Income ($)")
        plt.ylabel("Debt-to-Income Ratio (DTI)")
        plt.legend(["Non-Default (0)", "Default (1)"], loc="upper right")
        plt.savefig(os.path.join(output_dir, "dti_vs_annual_inc.png"), dpi=300)
        plt.close()

    print(f"\n[OK] EDA complete! All visual plots saved to: {output_dir}")
    return df_clean


if __name__ == "__main__":
    perform_eda()
