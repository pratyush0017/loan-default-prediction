"""
app.py
------
Streamlit Interactive Dashboard and Real-time Loan Default Prediction Application.
Built using 100% Python and Streamlit.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import predict_single_loan

# Set page configuration
st.set_page_config(
    page_title="Loan Default Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 0.5rem;
        color: #991B1B;
        font-weight: 600;
    }
    .risk-low {
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        padding: 1.2rem;
        border-radius: 0.5rem;
        color: #166534;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_overview():
    """Loads dataset summaries and metrics."""
    train_path = os.path.join(PROJECT_ROOT, "data", "loan_train.csv")
    fi_path = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.csv")
    
    df_train = pd.read_csv(train_path) if os.path.exists(train_path) else None
    df_fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
    return df_train, df_fi


def main():
    st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Application Section:",
        ["🎯 Real-Time Loan Prediction", "📊 Exploratory Data Dashboard", "🏆 Model Performance & Evaluation", "📖 Project Documentation"]
    )

    df_train, df_fi = load_data_overview()

    # =========================================================================
    # 1. REAL-TIME LOAN PREDICTION PAGE
    # =========================================================================
    if app_mode == "🎯 Real-Time Loan Prediction":
        st.markdown('<div class="main-header">🎯 Real-Time Loan Default Risk Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Assess credit risk and estimate the probability of borrower default before loan sanctioning.</div>', unsafe_allow_html=True)

        model_bundle_path = os.path.join(PROJECT_ROOT, "models", "final_model.joblib")
        if not os.path.exists(model_bundle_path):
            st.error(f"⚠️ Model bundle not found at `{model_bundle_path}`. Please run `python src/train.py` first.")
            return

        with st.form("loan_prediction_form"):
            st.subheader("1. Loan Details")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, max_value=50000.0, value=12000.0, step=500.0)
            with col2:
                funded_amnt = st.number_input("Funded Amount ($)", min_value=500.0, max_value=50000.0, value=12000.0, step=500.0)
            with col3:
                term = st.selectbox("Loan Term (Months)", [36, 60], index=0)
            with col4:
                int_rate = st.number_input("Interest Rate (%)", min_value=4.0, max_value=35.0, value=12.5, step=0.1)

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                installment = st.number_input("Monthly Installment ($)", min_value=15.0, max_value=2000.0, value=350.0, step=10.0)
            with col6:
                grade = st.selectbox("Credit Grade", ["A", "B", "C", "D", "E", "F", "G"], index=1)
            with col7:
                sub_grade = st.selectbox("Sub-Grade", [
                    f"{g}{i}" for g in ["A", "B", "C", "D", "E", "F", "G"] for i in range(1, 6)
                ], index=6)
            with col8:
                purpose = st.selectbox("Loan Purpose", [
                    "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
                    "small_business", "car", "wedding", "medical", "moving", "vacation", "house", "other"
                ], index=0)

            st.subheader("2. Borrower Financial & Employment Profile")
            col9, col10, col11, col12 = st.columns(4)
            with col9:
                annual_inc = st.number_input("Annual Income ($)", min_value=5000.0, max_value=1000000.0, value=65000.0, step=2500.0)
            with col10:
                emp_length = st.selectbox("Employment Length", [
                    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                    "6 years", "7 years", "8 years", "9 years", "10+ years"
                ], index=5)
            with col11:
                home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=0)
            with col12:
                verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"], index=0)

            st.subheader("3. Credit History & Risk Metrics")
            col13, col14, col15, col16 = st.columns(4)
            with col13:
                dti = st.number_input("Debt-to-Income (DTI %)", min_value=0.0, max_value=50.0, value=15.2, step=0.5)
            with col14:
                inq_last_6mths = st.number_input("Inquiries in Last 6 Months", min_value=0, max_value=15, value=1, step=1)
            with col15:
                delinq_2yrs = st.number_input("Delinquencies in 2 Years", min_value=0, max_value=20, value=0, step=1)
            with col16:
                pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0, max_value=5, value=0, step=1)

            col17, col18, col19, col20 = st.columns(4)
            with col17:
                open_acc = st.number_input("Open Credit Lines", min_value=1, max_value=60, value=10, step=1)
            with col18:
                total_acc = st.number_input("Total Credit Accounts", min_value=1, max_value=100, value=22, step=1)
            with col19:
                revol_bal = st.number_input("Revolving Balance ($)", min_value=0.0, max_value=500000.0, value=9500.0, step=500.0)
            with col20:
                revol_util = st.number_input("Revolving Line Utilization (%)", min_value=0.0, max_value=100.0, value=48.5, step=1.0)

            col21, col22, col23, col24 = st.columns(4)
            with col21:
                addr_state = st.selectbox("Borrower State", [
                    "CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MA",
                    "MD", "AZ", "WA", "CO", "MN", "MO", "NV", "CT", "TN", "OR", "WI", "AL", "SC"
                ], index=0)
            with col22:
                issue_d = st.text_input("Issue Date (e.g. Dec-11)", value="Dec-11")
            with col23:
                earliest_cr_line = st.text_input("Earliest Credit Line (e.g. May-99)", value="May-99")
            with col24:
                pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0, step=1)

            submitted = st.form_submit_button("🔍 Predict Loan Default Risk", use_container_width=True, type="primary")

        if submitted:
            borrower_dict = {
                "loan_amnt": loan_amnt,
                "funded_amnt": funded_amnt,
                "funded_amnt_inv": funded_amnt,
                "term": term,
                "int_rate": int_rate,
                "installment": installment,
                "grade": grade,
                "sub_grade": sub_grade,
                "emp_length": emp_length,
                "home_ownership": home_ownership,
                "annual_inc": annual_inc,
                "verification_status": verification_status,
                "purpose": purpose,
                "addr_state": addr_state,
                "dti": dti,
                "delinq_2yrs": delinq_2yrs,
                "earliest_cr_line": earliest_cr_line,
                "issue_d": issue_d,
                "inq_last_6mths": inq_last_6mths,
                "mths_since_last_delinq": np.nan,
                "mths_since_last_record": np.nan,
                "open_acc": open_acc,
                "pub_rec": pub_rec,
                "revol_bal": revol_bal,
                "revol_util": revol_util,
                "total_acc": total_acc,
                "pub_rec_bankruptcies": pub_rec_bankruptcies
            }

            try:
                res = predict_single_loan(borrower_dict, model_bundle_path)
                
                st.markdown("### 📋 Credit Decision & Assessment Result")
                r_col1, r_col2, r_col3 = st.columns([1.2, 1, 1])

                with r_col1:
                    if res["predicted_class"] == 1:
                        st.markdown(f"""
                        <div class="risk-high">
                            <h3>⚠️ HIGH DEFAULT RISK</h3>
                            <p>Prediction: <strong>Likely to Default (1)</strong></p>
                            <p>Recommendation: <strong>Reject or Require Additional Collateral/Guarantor</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-low">
                            <h3>✅ LOW DEFAULT RISK</h3>
                            <p>Prediction: <strong>Likely to Fully Pay (0)</strong></p>
                            <p>Recommendation: <strong>Approve Loan at Standard Terms</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                with r_col2:
                    st.metric("Probability of Default", f"{res['default_probability']*100:.2f}%")
                    st.progress(res["default_probability"])

                with r_col3:
                    st.metric("Probability of Full Payment", f"{res['non_default_probability']*100:.2f}%")
                    st.progress(res["non_default_probability"])

                # Feature contribution note
                st.info(f"💡 **Key Risk Drivers for this profile**: Grade ({grade}), Interest Rate ({int_rate}%), DTI ({dti}%), and Annual Income (${annual_inc:,.0f}).")

            except Exception as e:
                st.error(f"Error executing prediction: {str(e)}")

    # =========================================================================
    # 2. EXPLORATORY DATA DASHBOARD
    # =========================================================================
    elif app_mode == "📊 Exploratory Data Dashboard":
        st.markdown('<div class="main-header">📊 Exploratory Data Analysis (EDA) Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive insights, statistical distributions, and risk patterns across 27,000+ loan records.</div>', unsafe_allow_html=True)

        if df_train is not None:
            # Key Metric KPIs
            total_loans = len(df_train)
            defaults = (df_train["loan_status"] == 1).sum()
            default_rate = (defaults / total_loans) * 100
            avg_loan = df_train["loan_amnt"].mean()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Loan Records", f"{total_loans:,}")
            kpi2.metric("Overall Default Rate", f"{default_rate:.2f}%")
            kpi3.metric("Average Loan Amount", f"${avg_loan:,.2f}")
            kpi4.metric("Non-Default Loans", f"{total_loans - defaults:,}")

            st.markdown("---")

        # Visualizations Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target & Risk Distributions", "📈 Continuous Features", "🗂️ Categorical Breakdowns", "🔥 Correlation Heatmap"])

        plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                t_plot = os.path.join(plots_dir, "target_distribution.png")
                if os.path.exists(t_plot):
                    st.image(t_plot, caption="Target Distribution (Binary Class Imbalance)", use_container_width=True)
            with col_b:
                g_plot = os.path.join(plots_dir, "grade_vs_default.png")
                if os.path.exists(g_plot):
                    st.image(g_plot, caption="Default Probability Escalation by Credit Grade", use_container_width=True)

        with tab2:
            num_plot = os.path.join(plots_dir, "numerical_distributions.png")
            if os.path.exists(num_plot):
                st.image(num_plot, caption="Distributions of Key Continuous Loan Attributes", use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                ir_plot = os.path.join(plots_dir, "int_rate_by_status.png")
                if os.path.exists(ir_plot):
                    st.image(ir_plot, caption="Interest Rate by Loan Status", use_container_width=True)
            with col_d:
                dti_plot = os.path.join(plots_dir, "dti_vs_annual_inc.png")
                if os.path.exists(dti_plot):
                    st.image(dti_plot, caption="DTI vs Annual Income Relationship", use_container_width=True)

        with tab3:
            cat_plot = os.path.join(plots_dir, "categorical_distributions.png")
            if os.path.exists(cat_plot):
                st.image(cat_plot, caption="Categorical Feature Breakdown by Default Status", use_container_width=True)

        with tab4:
            corr_plot = os.path.join(plots_dir, "correlation_heatmap.png")
            if os.path.exists(corr_plot):
                st.image(corr_plot, caption="Feature Correlation Heatmap with Target", use_container_width=True)

    # =========================================================================
    # 3. MODEL PERFORMANCE & EVALUATION
    # =========================================================================
    elif app_mode == "🏆 Model Performance & Evaluation":
        st.markdown('<div class="main-header">🏆 Model Performance & Evaluation Benchmark</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Validation comparison across Logistic Regression, Random Forest, and CatBoost.</div>', unsafe_allow_html=True)

        # Actual Model Performance Table
        perf_data = {
            "Model": ["Logistic Regression (Balanced)", "Random Forest (Balanced)", "CatBoost Classifier (Champion)"],
            "Accuracy": [0.6645, 0.6936, 0.6956],
            "Precision": [0.2481, 0.2582, 0.2651],
            "Recall": [0.6302, 0.5774, 0.6025],
            "F1 Score": [0.3561, 0.3568, 0.3682],
            "ROC-AUC": [0.7138, 0.7065, 0.7227]
        }
        perf_df = pd.DataFrame(perf_data).set_index("Model")
        st.dataframe(perf_df.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True)

        st.markdown("---")
        plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            roc_plot = os.path.join(plots_dir, "roc_curves.png")
            if os.path.exists(roc_plot):
                st.image(roc_plot, caption="Validation ROC Curves (CatBoost AUC = 0.7227)", use_container_width=True)
        with col_m2:
            pr_plot = os.path.join(plots_dir, "pr_curves.png")
            if os.path.exists(pr_plot):
                st.image(pr_plot, caption="Validation Precision-Recall Curves", use_container_width=True)

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            cm_plot = os.path.join(plots_dir, "confusion_matrices.png")
            if os.path.exists(cm_plot):
                st.image(cm_plot, caption="Validation Confusion Matrices", use_container_width=True)
        with col_m4:
            comp_plot = os.path.join(plots_dir, "model_comparison.png")
            if os.path.exists(comp_plot):
                st.image(comp_plot, caption="Comprehensive Metric Comparison Bar Chart", use_container_width=True)

        st.subheader("🔥 Top Predictive Features (CatBoost Importance)")
        col_fi1, col_fi2 = st.columns([1.2, 1])
        with col_fi1:
            fi_plot = os.path.join(plots_dir, "feature_importance.png")
            if os.path.exists(fi_plot):
                st.image(fi_plot, caption="Top 20 Features Driving Default Risk", use_container_width=True)
        with col_fi2:
            if df_fi is not None:
                st.dataframe(df_fi.head(15), use_container_width=True)

    # =========================================================================
    # 4. PROJECT DOCUMENTATION
    # =========================================================================
    elif app_mode == "📖 Project Documentation":
        st.markdown('<div class="main-header">📖 Loan Default Prediction - Project Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Executive Summary
        This project builds an end-to-end Machine Learning system to predict borrower loan default (`loan_status` = 1) using financial, demographic, and credit risk indicators.

        ### Key Technical Components
        1. **Data Cleaning & Anti-Leakage Controls**:
           - Explicitly stripped 12 post-loan repayment & collection features (`out_prncp`, `total_pymnt`, `recoveries`, `last_pymnt_amnt`, etc.) to prevent target leakage.
           - Transformed string percentages (`int_rate`, `revol_util`) and employment lengths to pure numeric values.
           - Extracted vintage and maturity features (`issue_year`, `issue_month`, `credit_history_year`).
        2. **Multi-Model Comparison**:
           - Evaluated **Logistic Regression**, **Random Forest**, and **CatBoost Classifier**.
           - Addressed 85:15 class imbalance through balanced loss weighting and ROC-AUC evaluation.
           - Selected **CatBoost** (Validation ROC-AUC: **0.7227**) as the champion model.
        3. **Production Inference & App**:
           - 100% Python-based backend and UI using Streamlit.
           - Zero JavaScript / Node dependencies.
        """)


if __name__ == "__main__":
    main()

