# credit-card-fraud-detection/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

# -----------------------------
# Download and load models
# -----------------------------

model_url = "https://github.com/shubhumre777/gsoc-ml-projects/raw/main/credit-card-fraud-detection/models/rf_creditcard_model.pkl"
scaler_url = "https://github.com/shubhumre777/gsoc-ml-projects/raw/main/credit-card-fraud-detection/models/scaler.pkl"

@st.cache_data(show_spinner=False)
def load_model(url):
    response = requests.get(url)
    return joblib.load(BytesIO(response.content))

rf_model = load_model(model_url)
scaler = load_model(scaler_url)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("Credit Card Fraud Detection App")

st.write("""
Enter the transaction details below to predict whether it is fraudulent.
""")

# Create input fields dynamically for 30 features
input_data = {}
for i in range(1, 29+1):  # V1 to V28
    input_data[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

input_data["Amount"] = st.number_input("Transaction Amount", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale features
scaled_df = scaler.transform(input_df)

# Prediction
if st.button("Predict"):
    pred = rf_model.predict(scaled_df)[0]
    pred_prob = rf_model.predict_proba(scaled_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction! (Probability: {pred_prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {pred_prob:.2f})")
