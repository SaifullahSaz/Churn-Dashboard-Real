import pandas as pd
import numpy as np
import joblib
import pickle
import re
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from supabase import create_client
import streamlit as st
import time

# Initialize Supabase client
def _get_supabase_client():
    url = st.secrets["supabase"]["SUPABASE_URL"]
    key = st.secrets["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

@st.cache_data(ttl=60*5)  # cache 5 minutes; adjust as needed
def fetch_table_from_supabase(table_name: str, filters: dict | None = None, limit: int = 1000):
    """
    Fetch rows from a Supabase table into a pandas DataFrame.
    - table_name: table to query
    - filters: dict of column->value for equality filters (simple)
    - limit: page size for each request (use moderate default)
    Returns pandas.DataFrame.
    """
    supabase = _get_supabase_client()

    rows = []
    start = 0
    page_size = limit

    while True:
        query = supabase.table(table_name).select("*").range(start, start + page_size - 1)
        # apply simple equality filters
        if filters:
            for col, val in filters.items():
                if val is None:
                    continue
                query = query.eq(col, val)

        result = query.execute()
        # Newer versions of the Supabase/PostgREST client may return different
        # response shapes. Be defensive: check for an 'error' attribute, then
        # fall back to status_code checks and include data in the message.
        err = getattr(result, "error", None)
        status = getattr(result, "status_code", None)
        if err or (status is not None and int(status) >= 400):
            if err:
                msg = getattr(err, "message", str(err))
            else:
                # Try to include useful response body
                body = getattr(result, "data", None)
                msg = str(body) if body is not None else f"status_code={status}"
            raise Exception(f"Supabase query failed: {msg}")

        page_data = result.data or []
        rows.extend(page_data)
        if len(page_data) < page_size:
            break
        start += page_size
        # small delay to be polite with rate limits
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Optional: normalize JSON/JSONB columns if any (expand nested dicts)
    # df = pd.json_normalize(rows)
    return df


# def upsert_predictions_to_supabase(table_name: str, df: pd.DataFrame, key_col: str = "customerID"):
#     """Upsert prediction rows back to Supabase in chunks.

#     df should contain the key_col to use for upsert conflict detection.
#     """
#     supabase = _get_supabase_client()
#     records = df.to_dict(orient="records")
#     chunk_size = 500
#     for i in range(0, len(records), chunk_size):
#         chunk = records[i : i + chunk_size]
#         res = supabase.table(table_name).upsert(chunk, on_conflict=key_col).execute()
#         # Defensive response handling — don't assume `.error` exists on the
#         # returned object. Check for error/status_code and include response
#         # data for diagnostics.
#         rerr = getattr(res, "error", None)
#         rstatus = getattr(res, "status_code", None)
#         if rerr or (rstatus is not None and int(rstatus) >= 400):
#             if rerr:
#                 msg = getattr(rerr, "message", str(rerr))
#             else:
#                 body = getattr(res, "data", None)
#                 msg = str(body) if body is not None else f"status_code={rstatus}"
#             raise Exception(f"Supabase upsert failed: {msg}")

def upsert_predictions_to_supabase(table_name, df):
    supabase = _get_supabase_client()

    # Columns that must exist
    required_cols = [
        "customerID", "MonthlyCharges", "TotalCharges", "tenure",
        "Contract", "churn_probability", "churn_label", "risk_level"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Helper: clean numeric fields safely
    def clean_float(val):
        if val in [None, "", " ", "  ", np.nan]:
            return None
        try:
            return float(val)
        except:
            return None  # fallback instead of crash

    records = []
    for _, row in df[required_cols].iterrows():
        rec = {
            "customerid": row["customerID"],
            "monthlycharges": clean_float(row["MonthlyCharges"]),
            "totalcharges": clean_float(row["TotalCharges"]),
            "tenure": clean_float(row["tenure"]),
            "contract": row["Contract"] if row["Contract"] not in ["", " ", None] else None,
            "churn_probability": clean_float(row["churn_probability"]),
            "churn_label": int(row["churn_label"]) if str(row["churn_label"]).isdigit() else None,
            "risk_level": row["risk_level"],
        }
        records.append(rec)

    for rec in records:
        response = supabase.table(table_name).insert(rec).execute()

        # Handle Supabase errors
        if hasattr(response, "error") and response.error:
            raise Exception(response.error.message)

    return True


# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration if the consuming app hasn't configured logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Module-level cache for loaded artifacts
MODEL = None
FEATURE_LIST = None
SCALER = None


def _load_artifacts():
    """Load model, feature list and scaler from `models/best_model.pkl` if present.

    Returns (model, feature_list, scaler_or_None)
    Caches results in module-level variables MODEL, FEATURE_LIST, SCALER.
    """
    global MODEL, FEATURE_LIST, SCALER
    if MODEL is not None or FEATURE_LIST is not None or SCALER is not None:
        return MODEL, FEATURE_LIST, SCALER

    p = "models/best_model.pkl"
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        # propagate as a clear error
        logger.exception("Failed to load model artifacts")
        raise Exception(f"Could not load '{p}': {e}")

    if isinstance(data, dict):
        MODEL = data.get("model") or data.get("estimator") or data.get("pipeline")
        FEATURE_LIST = data.get("features") or data.get("feature_list")
        SCALER = data.get("scaler") or data.get("preprocessor")
        logger.info("Loaded artifacts from %s: model=%s, features=%s, scaler=%s", p, type(MODEL).__name__ if MODEL is not None else None, 'present' if FEATURE_LIST else None, 'present' if SCALER else None)
    else:
        # Unknown structure: try best-effort
        MODEL = data
        FEATURE_LIST = None
        SCALER = None

    logger.debug("Artifact load complete: MODEL=%s FEATURE_LIST=%s SCALER=%s", type(MODEL).__name__ if MODEL is not None else None, FEATURE_LIST if FEATURE_LIST is not None else None, type(SCALER).__name__ if SCALER is not None else None)

    return MODEL, FEATURE_LIST, SCALER


def load_model():
    """Backward-compatible loader: returns (model, feature_list).

    Internally uses `_load_artifacts()` which may also populate `SCALER`.
    """
    model, features, scaler = _load_artifacts()
    return model, features


def preprocess_data(df):
    """Cleans and preprocesses uploaded dataset for prediction."""
    
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scale numerical columns (use same columns as in training if scaler available)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    present_num_cols = [c for c in num_cols if c in df.columns]
    if present_num_cols:
        # Try to use a persisted scaler if available
        try:
            # Ensure artifacts loaded
            _load_artifacts()
        except Exception:
            # No persisted artifacts available; fall back to fitting a local scaler
            warnings.warn("Persisted scaler not found; scaling using a locally-fitted StandardScaler.")

        if SCALER is not None:
            try:
                df[present_num_cols] = SCALER.transform(df[present_num_cols])
            except Exception:
                # If the scaler can't transform due to shape mismatch, fall back
                warnings.warn("Saved scaler couldn't transform data; falling back to local scaling.")
                local_scaler = StandardScaler()
                df[present_num_cols] = local_scaler.fit_transform(df[present_num_cols])
        else:
            local_scaler = StandardScaler()
            df[present_num_cols] = local_scaler.fit_transform(df[present_num_cols])
    
    # Derived features
    if 'tenure' in df.columns and 'TotalCharges' in df.columns:
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df


# def predict_df(df, model=None):
#     """Take an input dataframe (raw uploaded), preprocess, align features with the
#     training feature list, run the model and return the original dataframe with
#     added columns: 'Churn_Probability', 'Churn_Binary', 'Predicted_Churn'.

#     Note: this function will call `load_model()` to obtain both the model and
#     the expected `feature_list` saved at training time.
#     """
#     # Keep a copy of original dataframe so we don't remove identifying cols like customerID
#     original = df.copy()

#     # Load model and expected features
#     try:
#         model_obj, feature_list = load_model()
#     except Exception:
#         raise

#     # Preprocess a copy for prediction (preprocess_data may drop customerID)
#     processed = preprocess_data(df.copy())

#     # Ensure all expected features exist; add missing with zeros
#     if feature_list is not None:
#         for col in feature_list:
#             if col not in processed.columns:
#                 processed[col] = 0

#         # Reorder columns to match training
#         processed = processed[feature_list]

#     # Predict probabilities (try predict_proba, fall back to predict)
#     try:
#         probs = model_obj.predict_proba(processed)[:, 1]
#     except Exception:
#         # Not all models implement predict_proba; fall back to predict
#         preds = model_obj.predict(processed)
#         # If predictions are 0/1 class labels, map them to 0/1 probabilities
#         probs = preds.astype(float)

#     # Attach predictions to the original dataframe
#     original["Churn_Probability"] = probs
#     original["Churn_Binary"] = (original["Churn_Probability"] >= 0.5).astype(int)
#     original["Predicted_Churn"] = original["Churn_Binary"].map({0: "No Churn", 1: "Churn"})

#     return original

def predict_df(df, return_processed=False):
    """
    Preprocess input df, run predictions, and return:
    - original df with predictions
    - AND optionally the processed df for SHAP
    """
    original = df.copy()

    # Load model + feature list
    model_obj, feature_list = load_model()

    # Preprocess
    processed = preprocess_data(df.copy())

    # Ensure model-required columns exist
    if feature_list is not None:
        for col in feature_list:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[feature_list]

    # Predict probabilities
    try:
        probs = model_obj.predict_proba(processed)[:, 1]
    except Exception:
        preds = model_obj.predict(processed)
        probs = preds.astype(float)

    # Add predictions
    original["churn_probability"] = probs
    original["churn_label"] = (probs >= 0.5).astype(int)

    # Risk segmentation
    def risk_map(p):
        if p >= 0.8:
            return "Critical"
        elif p >= 0.6:
            return "High"
        elif p >= 0.4:
            return "Medium"
        elif p >= 0.2:
            return "Low"
        else:
            return "Very Low"

    original["risk_level"] = original["churn_probability"].apply(risk_map)

    # Return both raw+predictions and processed DF if requested
    if return_processed:
        return original, processed
    else:
        return original




def load_metrics_from_notebook(nb_path="model_training.ipynb"):
    """Try to extract the printed metrics table from the training notebook.

    This is a best-effort parser: it reads the notebook as text and looks for
    lines containing model names and five numeric columns: Accuracy, Precision,
    Recall, F1, ROC-AUC. Returns a pandas.DataFrame if any rows are found,
    """
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            txt = f.read()
    except FileNotFoundError:
        logger.info("Notebook %s not found, cannot extract metrics.", nb_path)
        return None

    # Regex to find lines like: "Logistic Regression  0.792883   0.595016  0.542614  0.567608  0.846356"
    pattern = re.compile(r"^(?P<model>[A-Za-z0-9 _\-]+?)\s+(?P<accuracy>\d+\.\d+)\s+(?P<precision>\d+\.\d+)\s+(?P<recall>\d+\.\d+)\s+(?P<f1>\d+\.\d+)\s+(?P<roc_auc>\d+\.\d+)", re.MULTILINE)

    rows = []
    # First attempt: search the raw notebook JSON text (covers some notebooks that contain outputs inline)
    for m in pattern.finditer(txt):
        rows.append({
            "Model": m.group("model").strip(),
            "Accuracy": float(m.group("accuracy")),
            "Precision": float(m.group("precision")),
            "Recall": float(m.group("recall")),
            "F1": float(m.group("f1")),
            "ROC-AUC": float(m.group("roc_auc")),
        })

    # Second attempt: if no rows found, parse the notebook JSON and look into cell outputs
    if not rows:
        try:
            import json

            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            for cell in nb.get("cells", []):
                for output in cell.get("outputs", []) or []:
                    # outputs may have 'text' or 'data' with 'text/plain'
                    text = None
                    if output.get("output_type") == "stream":
                        text = "".join(output.get("text") or [])
                    else:
                        data = output.get("data") or {}
                        text = data.get("text/plain") or data.get("text") or None
                        if isinstance(text, list):
                            text = "".join(text)

                    if not text:
                        continue

                    for m in pattern.finditer(text):
                        rows.append({
                            "Model": m.group("model").strip(),
                            "Accuracy": float(m.group("accuracy")),
                            "Precision": float(m.group("precision")),
                            "Recall": float(m.group("recall")),
                            "F1": float(m.group("f1")),
                            "ROC-AUC": float(m.group("roc_auc")),
                        })
        except Exception:
            logger.exception("Failed to parse notebook outputs for metrics")

    if not rows:
        logger.info("No metrics rows found in %s", nb_path)
        return None

    df = pd.DataFrame(rows)
    # If the notebook printed the models in a certain order, preserve it.
    return df


