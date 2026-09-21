import streamlit as st
import joblib
import pandas as pd

# Load model and scaler

model = joblib.load('models/rf_creditcard_model.pkl')
scaler = joblib.load('models/scaler.pkl')



# Page configuration


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)



# Title

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction details below to check whether "
    "the transaction is Normal or Fraudulent."
)

# Feature list

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



# Input form

inputs = {}

for feature in features:

    inputs[feature] = st.number_input(
        feature,
        value=0.0
    )


# Prediction

if st.button("🔍 Predict Transaction"):

    # Convert input to DataFrame
    input_df = pd.DataFrame([inputs])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Result
    if prediction == 0:

        st.success(
            "✅ Transaction is Normal"
        )

    else:

        st.error(
            "⚠️ Transaction is Fraudulent"
        )
