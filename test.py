import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="Earnings Manipulation Detection",
    layout="centered"
)

st.title("📊 Earnings Manipulation Detection (XGBoost)")
st.markdown("Detect potential earnings manipulation using **Beneish ratios + XGBoost**")

# ----------------------------------
# Load data
# ----------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Earnings Manipulator (1).xlsx")
    df["Manipulator"] = df["Manipulator"].map({"No": 0, "Yes": 1})
    return df

df = load_data()

features = ['DSRI','GMI','AQI','SGI','DEPI','SGAI','ACCR','LEVI']
X = df[features]
y = df["Manipulator"]

# ----------------------------------
# Train model
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------
# Model performance
# ----------------------------------
y_pred = model.predict(X_test)
st.subheader("📈 Model Performance")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
st.write("F1 Score:", round(f1_score(y_test, y_pred), 3))

# ----------------------------------
# User input section
# ----------------------------------
st.subheader("🧮 Enter Financial Ratios")

def user_input():
    DSRI = st.number_input("DSRI", value=1.0)
    GMI  = st.number_input("GMI", value=1.0)
    AQI  = st.number_input("AQI", value=1.0)
    SGI  = st.number_input("SGI", value=1.0)
    DEPI = st.number_input("DEPI", value=1.0)
    SGAI = st.number_input("SGAI", value=1.0)
    ACCR = st.number_input("ACCR", value=0.0)
    LEVI = st.number_input("LEVI", value=1.0)

    data = {
        "DSRI": DSRI,
        "GMI": GMI,
        "AQI": AQI,
        "SGI": SGI,
        "DEPI": DEPI,
        "SGAI": SGAI,
        "ACCR": ACCR,
        "LEVI": LEVI
    }
    return pd.DataFrame([data])

input_df = user_input()

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Manipulation Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Likely Earnings Manipulator | Probability: {probability:.2%}")
    else:
        st.success(f"Not Likely a Manipulator | Probability: {probability:.2%}")

# ----------------------------------
# Feature importance
# ----------------------------------
st.subheader("📌 Feature Importance (XGBoost)")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))
