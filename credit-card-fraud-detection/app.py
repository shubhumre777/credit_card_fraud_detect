import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction values and predict whether it is **Fraudulent** or **Normal**")

# Loadinf the  model and the scaler
model = joblib.load("models/rf_creditcard_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Input section
st.subheader("Transaction Details")

inputs = {}
feature_names = [
    'Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
    'V10','V11','V12','V13','V14','V15','V16','V17','V18','V19',
    'V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'
]

for feature in feature_names:
    inputs[feature] = st.number_input(feature, value=0.0)

# press button to predict : 

if st.button("🔍 Predict"):
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # predicts whether its a fraud or not ??????
    
    if prediction == 0:
        st.success("✅ Transaction is NORMAL")
    else:
        st.error("🚨 FRAUDULENT Transaction Detected!")
