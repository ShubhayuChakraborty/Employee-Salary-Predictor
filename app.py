import streamlit as st
import joblib
import numpy as np

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# ---------------------- SIDEBAR ----------------------
st.sidebar.header("🔧 Model Settings")
st.sidebar.info("This app predicts the salary based on experience and job rating.")

# ---------------------- MAIN TITLE ----------------------
st.title("💼 Salary Prediction App")
st.write("Estimate the salary of an employee based on their experience and job rating.")

st.divider()

# ---------------------- INPUT FIELDS ----------------------
years = st.number_input(
    "🧑‍💻 Years of Experience",
    value=1,
    step=1,
    min_value=0,
)

jobrate = st.number_input(
    "⭐ Job Rating (0 - 5)",
    value=3.5,
    step=0.5,
    min_value=0.0,
    max_value=5.0
)

# ---------------------- LOAD MODEL ----------------------
try:
    model = joblib.load("linearmodel.pkl")
except:
    st.error("❌ Model file not found! Please place `linearmodel.pkl` in the project folder.")
    st.stop()

st.divider()

# ---------------------- PREDICTION BUTTON ----------------------
if st.button("🚀 Predict Salary"):
    st.balloons()
    x = np.array([[years, jobrate]])
    
    try:
        prediction = model.predict(x)[0]
        st.success(f"### 💰 Predicted Salary: **₹{prediction:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("👆 Enter values and click **Predict Salary**")

