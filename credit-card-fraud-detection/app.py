import streamlit as st
import joblib
import pandas as pd
import requests
from io import BytesIO


st.title("Credit Card Fraud Detection")
st.write("Enter transaction details to predict if it is Fraudulent or Normal")

st.markdown("""
<div style="
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    margin: 30px 0;
"></div>
""", unsafe_allow_html=True)


url_model = "https://github.com/shubhumre777/gsoc-ml-projects/raw/main/credit-card-fraud-detection/models/rf_creditcard_model.pkl"
url_scaler = "https://github.com/shubhumre777/gsoc-ml-projects/raw/main/credit-card-fraud-detection/models/scaler.pkl"

model = joblib.load(BytesIO(requests.get(url_model).content))
scaler = joblib.load(BytesIO(requests.get(url_scaler).content))

features = [
    'Time',
    'V1', 'V2', 'V3', 'V4', 'V5',
    'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15',
    'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25',
    'V26', 'V27', 'V28',
    'Amount'
]

inputs = {}

for col in features:
    inputs[col] = st.number_input(
        f'{col}',
        value=0.0
    )

if st.button('Predict'):

    input_df = pd.DataFrame([inputs])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    fraud_probability = probabilities[1]

    fraud_percentage = fraud_probability * 100

    risk_score = round(fraud_percentage)

    if risk_score < 20:
        risk_level = "🟢 Low Risk"
    elif risk_score < 50:
        risk_level = "🟡 Medium Risk"
    elif risk_score < 80:
        risk_level = "🟠 High Risk"
    else:
        risk_level = "🔴 Critical Risk"

    st.markdown("""
    <div style="
        margin-top: 30px;
        margin-bottom: 25px;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        background: linear-gradient(90deg, #111827, #1e293b);
        border: 1px solid #334155;
    ">
        <h3 style="margin: 0; color: #38bdf8;">
            🛡️ TRANSACTION RISK ANALYSIS
        </h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{fraud_percentage:.2f}%"
        )

    with col2:
        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    with col3:
        st.metric(
            "Risk Level",
            risk_level
        )
    

    # st.metric(
    #     "Fraud Probability",
    #     f"{fraud_percentage:.2f}%"
    # )

    st.progress(int(fraud_percentage))

    if prediction == 0:
        st.success("✅ Transaction is Normal")
    else:
        st.error("⚠️ Transaction is Fraudulent")
