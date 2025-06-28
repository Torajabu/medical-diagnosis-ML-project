# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load trained models
try:
    lr_model = joblib.load("lr_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
except FileNotFoundError:
    st.error("Model files not found! Please train and save models first.")
    st.stop()

# App title
st.title("🏥 Medical Disease Diagnosis using AI")
st.write("Enter patient information to predict disease risk")
st.markdown("**App made by Rajab**")

# Create input form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 30)
    sex = st.radio("Gender", ["Male", "Female"])
    bp = st.slider("Blood Pressure", 80, 200, 120)

with col2:
    chol = st.slider("Cholesterol Level", 100, 400, 200)
    sugar = st.slider("Sugar Level", 50, 300, 120)
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Random Forest"])

# Prepare input data
input_data = pd.DataFrame({
    "age": [age],
    "sex": [0 if sex == "Male" else 1],
    "blood_pressure": [bp],
    "cholesterol": [chol],
    "sugar": [sugar]
})

# Make prediction
if st.button("🔍 Diagnose"):
    with st.spinner("Analyzing..."):
        if model_choice == "Logistic Regression":
            pred = lr_model.predict(input_data)
            prob = lr_model.predict_proba(input_data)[0]
        else:
            pred = rf_model.predict(input_data)
            prob = rf_model.predict_proba(input_data)[0]
    
    # Display results
    result = "At Risk" if pred[0] == 1 else "Low Risk"
    confidence = max(prob) * 100
    
    if pred[0] == 1:
        st.error(f"⚠️ Result: {result} (Confidence: {confidence:.1f}%)")
        st.warning("Please consult with a healthcare professional for proper medical advice.")
    else:
        st.success(f"✅ Result: {result} (Confidence: {confidence:.1f}%)")
        st.info("Remember to maintain regular health checkups.")
    
    # Show input summary
    st.subheader("Input Summary:")
    st.write(f"Age: {age}, Gender: {sex}, BP: {bp}, Cholesterol: {chol}, Sugar: {sugar}")

# Add disclaimer
st.markdown("---")
st.markdown("**App made by Rajab**")
st.markdown("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")
