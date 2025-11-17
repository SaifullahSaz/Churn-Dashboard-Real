# %%
print("1. IMPORTING LIBRARIES")
# Basic
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Explainability
import shap


# %%
print("2. LOAD AND INSPECT DATASET")
# Load dataset (replace path if needed)
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Overview
print(df.shape)
df.head()
df.info()


# %%
print("3. DATA CLEANING")
# Remove customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Check duplicates
df.drop_duplicates(inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})


# %%
df.describe()

# %%
print("3.1 NUMERICAL SUMMARY STATISTICS")
from IPython.display import display
# Select numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("Numerical columns:", num_cols)

# Summary statistics with missing counts
num_stats = df[num_cols].describe().T
num_stats['missing'] = df[num_cols].isna().sum()
display(num_stats)

# Boxplots for each numerical column
import math
n = len(num_cols)
cols = 3
rows = math.ceil(n / cols)
plt.figure(figsize=(cols*5, rows*4))
for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.tight_layout()
plt.show()

# %%
print("4. ENCODING AND SCALING")
# Label encode categorical features
cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols] = scaler.fit_transform(df[num_cols])


# %%
print("5. FEATURE ENGINEERING (enhance with time-like logic)")
# Create tenure category
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['<1yr', '1-2yr', '2-4yr', '4-6yr'])

# Encode the new feature
df = pd.get_dummies(df, columns=['TenureGroup'], drop_first=True)

# Derived feature: average monthly spend per tenure
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

# Replace inf/nan
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

df.info()

# %%
print("6. FEATURE SELECTION")
# Feature importance using RandomForest
X = df.drop('Churn', axis=1)
y = df['Churn']

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(10)
print(top_features)

# Visualize
plt.figure(figsize=(10,6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features")
plt.show()


# %%
print("7. SPLIT DATA")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
print("8. MODEL TRAINING AND TUNING")
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

results_df = pd.DataFrame(results).T
print(results_df)


# %%
print("9. ROC Curve Visualization")
plt.figure(figsize=(8,6))
for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_proba):.2f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()


# %%
print("10. MODEL EXPLAINABILITY WITH SHAP")
explainer = shap.TreeExplainer(models["XGBoost"])
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")


# %%
print("11. FINDING THE BEST MODEL")
# Display results again
print("\nModel Performance Comparison:")
print(results_df.round(4))

# Find best model based on different metrics
print("\n" + "="*60)
print("Best Model by Different Metrics:")
print("="*60)
for metric in results_df.columns:
    best_model_name = results_df[metric].idxmax()
    best_score = results_df[metric].max()
    print(f"{metric:15s}: {best_model_name:20s} ({best_score:.4f})")

# Overall best model (using F1 score as primary metric for imbalanced data)
best_model_name = results_df['F1'].idxmax()
print("\n" + "="*60)
print(f"🏆 OVERALL BEST MODEL: {best_model_name}")
print("="*60)
print(f"\nPerformance metrics:")
print(results_df.loc[best_model_name].round(4))

# Store the best model
best_model = models[best_model_name]
print(f"\n✅ Best model stored in variable 'best_model'")

# %%
print("12. CONFUSION MATRIX FOR BEST MODEL")
# Get predictions for best model
y_pred_best = best_model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Print detailed metrics
print(f"\nConfusion Matrix for {best_model_name}:")
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# %%
import pickle
from sklearn.linear_model import LogisticRegression

# 1️⃣ Train your best model
best_model = LogisticRegression(max_iter=300)
best_model.fit(X, y)

# 2️⃣ Save the list of features used during training
feature_list = X.columns.tolist()

# 3️⃣ Store both model and features together in one file
model_data = {
    "model": best_model,
    "features": feature_list
}

# 4️⃣ Save using pickle (no need to also use joblib — keep it consistent)
import os
os.makedirs("models", exist_ok=True)

with open("models/best_model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("✅ Model and feature list saved successfully at models/best_model.pkl")


# %%
# Save cleaned dataframe to CSV
os.makedirs("data", exist_ok=True)
cleaned_file = "data/cleaned_telco_churn.csv"
df.to_csv(cleaned_file, index=False)
print(f"Saved cleaned dataframe ({df.shape[0]} rows, {df.shape[1]} columns) to: {cleaned_file}")

# %%
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

# Get required columns from X_test
required_cols = X_test.columns

# Create a sample dict with all columns, set reasonable defaults
sample_dict = {col: 0 for col in required_cols}
sample_dict["tenure"] = 12
sample_dict["MonthlyCharges"] = 70
sample_dict["TotalCharges"] = 840

# If you want to set some categorical features, e.g.:
sample_dict["gender_Male"] = True
sample_dict["Partner_Yes"] = True
sample_dict["PhoneService_Yes"] = True
sample_dict["Contract_One year"] = True
sample_dict["PaperlessBilling_Yes"] = True

# Calculate AvgMonthlySpend as in feature engineering
sample_dict["AvgMonthlySpend"] = sample_dict["TotalCharges"] / (sample_dict["tenure"] + 1)

# Create DataFrame and ensure correct column order
sample = pd.DataFrame([sample_dict], columns=required_cols)

# Scale numerical features using the same scaler
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
sample[num_cols] = scaler.transform(sample[num_cols])

pred = model.predict(sample)
print("Prediction:", pred)

# Save feature list during training to ensure consistency

# After training your model
import joblib

feature_list = X.columns.tolist()

# Save both model and feature list
joblib.dump((best_model, feature_list), "best_model.pkl")



