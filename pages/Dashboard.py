import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    predict_df,
    load_model,
    fetch_table_from_supabase,
    upsert_predictions_to_supabase
)

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# Page title
st.title("📊 Churn Prediction Dashboard")

# -------------------------------------------------------------
# HELPER: NORMALIZE SUPABASE COLUMN NAMES
# -------------------------------------------------------------
def normalize_supabase_columns(df):
    """Rename Supabase lowercase columns to match original prediction DF."""
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
    df = df.rename(columns=rename_map)
    return df


# -------------------------------------------------------------
# LOAD FROM SUPABASE (BUTTON)
# -------------------------------------------------------------
st.subheader("📦 Load Stored Predictions")

if st.button("Load Predictions from Supabase"):
    try:
        df_supabase = fetch_table_from_supabase("predictions")

        if df_supabase.empty:
            st.info("No stored predictions found in Supabase.")
        else:
            df_supabase = normalize_supabase_columns(df_supabase)
            st.session_state["predictions"] = df_supabase
            st.success("Loaded predictions from Supabase!")
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")


# -------------------------------------------------------------
# FILE UPLOAD (IF NO PREDICTIONS STORED)
# -------------------------------------------------------------
if "predictions" not in st.session_state:
    st.warning("⚠️ No predictions available. Upload a CSV to generate predictions.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            with st.spinner("Generating predictions..."):
                result_df = predict_df(df.copy())
                st.session_state["predictions"] = result_df
            st.success("Predictions generated successfully!")
        except Exception as e:
            st.error(f"Failed to generate predictions: {e}")
            st.stop()
    else:
        st.stop()  # Stop page here until predictions exist

# -------------------------------------------------------------
# MAIN DATAFRAME
# -------------------------------------------------------------
result_df = st.session_state["predictions"]

# -------------------------------------------------------------
# SAVE TO SUPABASE BUTTON
# -------------------------------------------------------------
st.subheader("💾 Save Predictions")

if st.button("Save Predictions to Supabase"):
    try:
        upsert_predictions_to_supabase("predictions", result_df)
        st.success("Predictions successfully saved to Supabase!")
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
# CHARTS
# -------------------------------------------------------------
st.markdown("---")
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# ========== CHART 1 — Risk Distribution ==========
with row1_col1:
    st.markdown("### 📊 Churn Risk Distribution")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["Low", "Medium-Low", "Medium", "High", "Critical"]

    result_df["Risk_Segment"] = pd.cut(
        result_df["churn_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    dist_df = result_df["Risk_Segment"].value_counts().sort_index().reset_index()
    dist_df.columns = ["Risk Level", "Count"]

    fig1 = px.bar(
        dist_df, x="Risk Level", y="Count",
        color="Risk Level",
        color_discrete_sequence=["#10b981", "#84cc16", "#f59e0b", "#f97316", "#ef4444"]
    )
    fig1.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=40))
    st.plotly_chart(fig1, use_container_width=True)

# ========== CHART 2 — Churn by Contract ==========
with row1_col2:
    st.markdown("### 📋 Churn by Contract Type")

    if "Contract" in result_df.columns:
        contract_stats = result_df.groupby("Contract").agg(
            churned=("churn_label", "sum"),
            total=("churn_label", "count")
        ).reset_index()

        contract_stats["Churn_Rate"] = contract_stats["churned"] / contract_stats["total"] * 100

        fig2 = px.bar(
            contract_stats,
            x="Contract",
            y="Churn_Rate",
            color="Churn_Rate",
            color_continuous_scale="Reds",
            text="Churn_Rate"
        )
        fig2.update_traces(texttemplate="%{text:.1f}%")
        fig2.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=40))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Contract column missing from predictions.")

# ========== CHART 3 — Key Drivers ==========
with row2_col1:
    st.markdown("### 🔍 Key Churn Drivers")

    numeric_features = result_df.select_dtypes(include=["number"]).columns
    numeric_features = [c for c in numeric_features if c not in ["churn_label", "churn_probability"]]

    corr_values = {
        col: abs(result_df[col].corr(result_df["churn_label"]))
        for col in numeric_features
    }

    corr_df = pd.DataFrame(list(corr_values.items()), columns=["Feature", "Correlation"])
    corr_df = corr_df.sort_values("Correlation", ascending=False).head(10)

    fig3 = px.bar(
        corr_df,
        y="Feature",
        x="Correlation",
        orientation="h",
        color="Correlation",
        color_continuous_scale="Purples"
    )
    fig3.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig3, use_container_width=True)

# ========== CHART 4 — Tenure Trend ==========
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

        tenure_stats["Churn_Rate"] = tenure_stats["churned"] / tenure_stats["total"] * 100

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=tenure_stats["Tenure_Group"],
            y=tenure_stats["Churn_Rate"],
            mode="lines+markers",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=8)
        ))
        fig4.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=10, b=40),
            yaxis_title="Churn Rate (%)",
            xaxis_title="Tenure Group"
        )
        st.plotly_chart(fig4, use_container_width=True)


# -------------------------------------------------------------
# END OF FILE
# -------------------------------------------------------------
