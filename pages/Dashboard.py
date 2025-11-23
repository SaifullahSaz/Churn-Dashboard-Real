import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import pickle
import matplotlib.pyplot as plt

from utils import (
    predict_df,
    load_model,
    fetch_table_from_supabase,
    upsert_predictions_to_supabase
)

# -------------------------------------------------------------
# LOAD BEST MODEL (LOGISTIC REGRESSION + FEATURE LIST)
# -------------------------------------------------------------
with open("models/best_model.pkl", "rb") as file:
    model_data = pickle.load(file)

lr_model = model_data["model"]
feature_list = model_data["features"]

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.title("📊 Churn Prediction Dashboard")

# -------------------------------------------------------------
# SUPABASE COLUMN NORMALIZER
# -------------------------------------------------------------
def normalize_supabase_columns(df):
    rename_map = {
        "customerid": "customerID",
        "monthlycharges": "MonthlyCharges",
        "totalcharges": "TotalCharges",
        "tenure": "tenure",
        "contract": "Contract",
        "churn_probability": "churn_probability",
        "churn_label": "churn_label",
        "risk_level": "risk_level",
    }
    return df.rename(columns=rename_map)

# -------------------------------------------------------------
# LOAD FROM SUPABASE
# -------------------------------------------------------------
st.subheader("📦 Load Stored Predictions")

if st.button("Load Predictions from Supabase"):
    try:
        df_supabase = fetch_table_from_supabase("predictions")

        if df_supabase.empty:
            st.info("No stored predictions found in Supabase.")
        else:
            df_supabase = normalize_supabase_columns(df_supabase)

            # Recreate processed features for SHAP
            result_df, processed_df = predict_df(df_supabase.copy(), return_processed=True)
            st.session_state["predictions"] = result_df
            st.session_state["processed_df"] = processed_df

            st.success("Loaded predictions from Supabase!")
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")

# -------------------------------------------------------------
# UPLOAD CSV IF NO STORED PREDICTIONS
# -------------------------------------------------------------
if "predictions" not in st.session_state:
    st.warning("⚠️ No predictions available. Upload a CSV to generate predictions.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            with st.spinner("Generating predictions..."):
                result_df, processed_df = predict_df(df.copy(), return_processed=True)
                st.session_state["predictions"] = result_df
                st.session_state["processed_df"] = processed_df

            st.success("Predictions generated successfully!")
        except Exception as e:
            st.error(f"Failed to generate predictions: {e}")
            st.stop()
    else:
        st.stop()

# -------------------------------------------------------------
# GET FINAL DATAFRAMES
# -------------------------------------------------------------
result_df = st.session_state["predictions"]
processed_df = st.session_state.get("processed_df", None)

# SAFETY CHECK: If processed_df was not stored, rebuild it
if processed_df is None:
    _, processed_df = predict_df(result_df.copy(), return_processed=True)
    st.session_state["processed_df"] = processed_df

# -------------------------------------------------------------
# SAVE TO SUPABASE BUTTON
# -------------------------------------------------------------
st.subheader("💾 Save Predictions")

if st.button("Save Predictions to Supabase"):
    try:
        upsert_predictions_to_supabase("predictions", result_df)
        st.success("Predictions successfully saved!")
    except Exception as e:
        st.error(f"Failed to save: {str(e)}")

# -------------------------------------------------------------
# TOP METRICS
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Overall Churn Metrics")

col1, col2, col3, col4 = st.columns(4)

total_customers = len(result_df)
churned = int(result_df["churn_label"].sum())
churn_rate = churned / total_customers * 100 if total_customers > 0 else 0

high_risk = result_df[result_df["churn_probability"] >= 0.7]
revenue_at_risk = high_risk["MonthlyCharges"].sum() if "MonthlyCharges" in result_df else 0

with col1:
    st.metric("Total Customers", f"{total_customers:,}")
with col2:
    st.metric("Predicted Churn", f"{churned:,}")
with col3:
    st.metric("Churn Rate", f"{churn_rate:.1f}%")
with col4:
    st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo")

# -------------------------------------------------------------
# CHART LAYOUT
# -------------------------------------------------------------
st.markdown("---")
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ========== CHART 1 — RISK DISTRIBUTION ==========
with row1_col1:
    st.markdown("### 📊 Churn Risk Distribution")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["Low", "Medium-Low", "Medium", "High", "Critical"]

    result_df["Risk_Segment"] = pd.cut(
        result_df["churn_probability"], bins=bins, labels=labels, include_lowest=True
    )

    dist_df = (
        result_df["Risk_Segment"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    dist_df.columns = ["Risk Level", "Count"]

    fig1 = px.bar(
        dist_df,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        color_discrete_sequence=["#10b981", "#84cc16", "#f59e0b", "#f97316", "#ef4444"]
    )
    fig1.update_layout(height=300)
    st.plotly_chart(fig1, use_container_width=True)

# ========== CHART 2 — CHURN BY CONTRACT ==========
with row1_col2:
    st.markdown("### 📋 Churn by Contract Type")

    if "Contract" in result_df.columns:
        contract_stats = result_df.groupby("Contract").agg(
            churned=("churn_label", "sum"),
            total=("churn_label", "count")
        ).reset_index()

        contract_stats["Churn_Rate"] = (
            contract_stats["churned"] / contract_stats["total"] * 100
        )

        fig2 = px.bar(
            contract_stats,
            x="Contract",
            y="Churn_Rate",
            color="Churn_Rate",
            color_continuous_scale="Reds",
            text="Churn_Rate"
        )
        fig2.update_traces(texttemplate="%{text:.1f}%")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Contract column missing.")

# ========== CHART 3 — SHAP KEY DRIVERS ==========
with row2_col1:
    st.markdown("### 🔍 Key Churn Drivers (SHAP Explainability)")

    shap_input = processed_df[feature_list]

    explainer = shap.KernelExplainer(lr_model.predict_proba, shap_input)
    shap_values = explainer.shap_values(shap_input)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], shap_input, plot_type="bar", show=False)
    st.pyplot(fig)

    st.caption("Top features influencing churn (Logistic Regression + SHAP).")

# ========== CHART 4 — TENURE TREND ==========
with row2_col2:
    st.markdown("### ⏳ Churn Trend by Tenure")

    if "tenure" in result_df.columns:
        result_df["Tenure_Group"] = pd.cut(
            result_df["tenure"],
            bins=[0, 12, 24, 36, 48, 1000],
            labels=["0-12 mo", "13-24 mo", "25-36 mo", "37-48 mo", "48+ mo"]
        )

        tenure_stats = result_df.groupby("Tenure_Group").agg(
            churned=("churn_label", "sum"),
            total=("churn_label", "count")
        ).reset_index()

        tenure_stats["Churn_Rate"] = (
            tenure_stats["churned"] / tenure_stats["total"] * 100
        )

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=tenure_stats["Tenure_Group"],
            y=tenure_stats["Churn_Rate"],
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8),
        ))
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)
