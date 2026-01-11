import streamlit as st
import joblib
import pandas as pd

# Load saved model and scaler
model = joblib.load('models/rf_creditcard_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it is Fraudulent or Normal")

# Define expected columns
expected_cols = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
                 'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
                 'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount']

# Create input form dynamically
inputs = {}
for col in expected_cols:
    inputs[col] = st.number_input(f'{col}', value=0.0)

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# Ensure column order matches exactly what the scaler expects
input_df = input_df[expected_cols]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button('Predict'):
    prediction = model.predict(input_scaled)[0]
    if prediction == 0:
        st.success("✅ Transaction is Normal")
    else:
        st.error("⚠️ Transaction is Fraudulent")
