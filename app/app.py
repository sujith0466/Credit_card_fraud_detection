import streamlit as st
import joblib
import numpy as np

model = joblib.load('../models/random_forest_model.pkl')
scaler = joblib.load('../models/scaler.pkl')  # Save your scaler separately if needed

st.title("💳 Credit Card Fraud Detection")

amount = st.number_input("Transaction Amount", value=100.0)
time = st.number_input("Transaction Time", value=20000)

# Example V features (real model would use all)
v_features = [st.number_input(f"V{i}", value=0.0) for i in range(1, 29)]

input_data = np.array([time, amount] + v_features).reshape(1, -1)
input_data[:, :2] = scaler.transform(input_data[:, :2])  # scale Time & Amount

if st.button("Check Fraud"):
    prediction = model.predict(input_data)
    result = "🚨 Fraud" if prediction[0] == 1 else "✅ Not Fraud"
    st.subheader(f"Prediction: {result}")
