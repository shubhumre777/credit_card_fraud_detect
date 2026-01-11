import streamlit as st
import joblib
import pandas as pd
import requests
from io import BytesIO

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it is Fraudulent or Normal")

# Load model and scaler from GitHub raw URLs
url_model = "https://github.com/shubhumre777/gsoc-ml-projects/raw/main/credit-card-fraud-detection/models/rf_creditcard_model.pkl"
url_scaler = "https://github.com/shubhumre777/gsoc-ml-projects/raw/main/credit-card-fraud-detection/models/scaler.pkl"

model = joblib.load(BytesIO(requests.get(url_model).content))
scaler = joblib.load(BytesIO(requests.get(url_scaler).content))

# Create input form
inputs = {}
for col in ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
            'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
            'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']:
    inputs[col] = st.number_input(f'{col}', value=0.0)

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)[0]

# Show result
if st.button('Predict'):
    if prediction == 0:
        st.success("✅ Transaction is Normal")
    else:
        st.error("⚠️ Transaction is Fraudulent")
