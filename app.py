
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

st.title("🏠 Home — Customer Churn Dashboard")
st.write("Welcome to the Customer Churn Prediction dashboard.")

st.markdown("---")

st.subheader("About This App")
st.write("""
This dashboard helps predict customer churn using machine learning. 

**Features:**
- 📤 Upload customer data and get churn predictions
- 📊 Analyze churn patterns and trends  
- 📈 View model evaluation metrics
""")

st.markdown("---")

st.info("👈 Use the sidebar menu to navigate between pages")
st.write("Technical note: Preprocessing and model-loading logic is in `utils.py`.")

