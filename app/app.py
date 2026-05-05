import streamlit as st
import pickle
import pandas as pd
import os
import sys

# Fix import path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

from features import create_features

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)


def encode_input(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df


# ---------------- UI ----------------

st.title("Customer Churn Prediction (US Telecom Dataset)")

st.info("Use realistic values !. Monthly Charges: 20–120, Tenure: 0–72 months")

st.write("Enter customer details:")

# 🔹 Controlled Inputs
tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.slider("Monthly Charges", min_value=20, max_value=120, value=70)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

total_charges = monthly_charges * tenure if tenure > 0 else 0
st.write(f"~Estimated Total Charges: {total_charges:.2f}")

# ---------------- Predict ----------------
if st.button("Predict"):

    # Hidden defaults (neutral)
    input_dict = {
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": tenure,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": internet_service,
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": tech_support,
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": contract,
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }

    input_df = pd.DataFrame([input_dict])

    # Feature engineering
    input_df = create_features(input_df)

    # Encoding
    input_df = encode_input(input_df, encoders)

    # Prediction
    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob > 0.5 else 0

    # Output
    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {prob:.2f}")

    if prediction == 1:
        st.error("Customer is likely to churn !")
    else:
        st.success("Customer is likely to stay !")