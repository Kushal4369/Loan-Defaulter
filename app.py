import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Default Prediction App", page_icon="💰", layout="centered")

# Title
st.title("💳 Bank Loan Default Prediction")
st.markdown("Enter customer details below to predict loan default risk:")

# Sidebar inputs
st.sidebar.header("📋 Input Customer Data")
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Monthly Income ($)", 1000, 50000, 4000)
family = st.select_slider("Family Size", options=[1, 2, 3, 4, 5], value=2)
CCAvg = st.sidebar.number_input("Credit Card Average Spending ($)")
education = st.sidebar.selectbox("Education Level", [1, 2, 3])  # 1: Undergrad, 2: Graduate, 3: Advanced/Professional
mortgage = st.sidebar.number_input("Mortgage Amount ($)", 0, 650, 0)
securities_account = st.sidebar.selectbox("Securities Account", [1, 0])
cd_account = st.sidebar.selectbox("CD Account", [1, 0])
online = st.sidebar.selectbox("Online Banking", [1, 0])
credit_card = st.sidebar.selectbox("Credit Card", [1, 0])


# Prepare data
data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Family": [family],
    "CCAvg": [CCAvg],
    'Education': [education],
    'Mortgage': [mortgage],
    'Securities Account': [securities_account],
    'CD Account': [cd_account],
    'Online': [online],
    'CreditCard': [credit_card],})

# Scale features
scaled_data = scaler.transform(data)

# Predict
prediction = model.predict(scaled_data)[0]
proba = model.predict_proba(scaled_data)[0][1]

# Display result
st.subheader("📊 Prediction Result")

if prediction == 1:
    st.error(f"❌ The customer is likely to DEFAULT (Probability: {proba:.2f})")
else:
    st.success(f"✅ The customer is likely to REPAY the loan (Probability: {proba:.2f})")

# Optional: Show raw data
with st.expander("Show processed input data"):
    st.write(data)
