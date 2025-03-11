import streamlit as st
import pickle
import numpy as np
import os

# Load trained model
model_path = "bankruptcydt-model.pkl"  # Update with your model path/name 
if not os.path.exists(model_path):
    st.error("Model file not found. Please check the file path.")
    st.stop()

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    st.write(f"Model loaded successfully: {type(model)}")  # Debugging output
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Bankruptcy Prediction App")
st.write("Enter the required details to predict BANKRUPTCY.")

# Input fields for features
financial_flexibility = st.number_input("Financial Flexibility", min_value=0.0, max_value=1.0, step=0.5)
credibility = st.number_input("Credibility", min_value=0.0, max_value=1.0, step=0.5)
competitiveness = st.number_input("Competitiveness", min_value=0.0, max_value=1.0, step=0.5)

# Predict function
def predict_bankruptcy():
    features = np.array([[financial_flexibility, credibility, competitiveness]], dtype=np.float64)
    try:
        prediction = model.predict(features)
        return "The Company may most probably go Bankrupt" if prediction[0] == 0 else "The Company may Not go Bankrupt"
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error in prediction"

if st.button("Predict"):
    result = predict_bankruptcy()
    st.write(f"### Prediction: {result}")
