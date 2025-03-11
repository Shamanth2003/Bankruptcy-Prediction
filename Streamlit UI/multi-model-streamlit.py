import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image

# Define model configurations
MODEL_CONFIGS = {
    "DT Model": {
        "file": "bankruptcydt-model.pkl",
        "features": [
            {"name": "Industry Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Management Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Financial Flexibility", "min": 0.0, "max": 2.0, "step": 0.5},
            {"name": "Credibility", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Competitiveness", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Operating Risk", "min": 0.0, "max": 1.0, "step": 0.5}
        ]
    },
    "LR Model": {
        "file": "bankruptcylr-model.pkl",
        "features": [
            {"name": "Industry Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Management Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Financial Flexibility", "min": 0.0, "max": 2.0, "step": 0.5},
            {"name": "Credibility", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Competitiveness", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Operating Risk", "min": 0.0, "max": 1.0, "step": 0.5}
        ]
    },
    "RF Model": {
        "file": "bankruptcyrf-model.pkl",
        "features": [
            {"name": "Industry Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Management Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Financial Flexibility", "min": 0.0, "max": 2.0, "step": 0.5},
            {"name": "Credibility", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Competitiveness", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Operating Risk", "min": 0.0, "max": 1.0, "step": 0.5}
        ]
    },
    "XGB Model": {
        "file": "bankruptcyxgb-model.pkl",
        "features": [
            {"name": "Industry Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Management Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Financial Flexibility", "min": 0.0, "max": 2.0, "step": 0.5},
            {"name": "Credibility", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Competitiveness", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Operating Risk", "min": 0.0, "max": 1.0, "step": 0.5}
        ]
    },
    "SVM Model": {
        "file": "bankruptcysvm-model.pkl",
        "features": [
            {"name": "Industry Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Management Risk", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Financial Flexibility", "min": 0.0, "max": 2.0, "step": 0.5},
            {"name": "Credibility", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Competitiveness", "min": 0.0, "max": 1.0, "step": 0.5},
            {"name": "Operating Risk", "min": 0.0, "max": 1.0, "step": 0.5}
        ]
    }
}

# Cache model loading
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main UI
st.title("Advanced Bankruptcy Prediction System")

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# Model selection
with col2:
    selected_model = st.selectbox(
        "Select Prediction Model",
        options=list(MODEL_CONFIGS.keys())
    )

    st.write("### Model Description")
    st.write(f"You are using {selected_model} with {len(MODEL_CONFIGS[selected_model]['features'])} features.")


# Load selected model
model_path = MODEL_CONFIGS[selected_model]["file"]
model = load_model(model_path)

if model:
    # Input fields for selected model
    st.write("### Enter Model Features")
    input_values = []
    
    # Create three columns for input fields
    cols = st.columns(3)
    for idx, var in enumerate(MODEL_CONFIGS[selected_model]["features"]):
        with cols[idx % 3]:
            value = st.number_input(
                var["name"],
                min_value=var["min"],
                max_value=var["max"],
                step=var["step"]
            )
            input_values.append(value)

    # Prediction function
    def predict_bankruptcy(features):
        try:
            prediction = model.predict(features)
            probability = model.predict_proba(features)
            return prediction[0], probability[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None, None

    # Predict button
    if st.button("Predict Bankruptcy"):
        features = np.array([input_values], dtype=np.float64)
        prediction, probability = predict_bankruptcy(features)
        
        if prediction is not None:
            st.write("### Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                result_text = "Based on your input the Company May Go Bankrupt" if prediction == 0 else "Based on your input the Company May Not Go Bankrupt"
                result_color = "red" if prediction == 0 else "green"
                st.markdown(f"<h3 style='color: {result_color};'>{result_text}</h3>", 
                          unsafe_allow_html=True)
            
            with result_col2:
                confidence = probability[1] if prediction == 1 else probability[0]
                st.metric("Confidence Level", f"{confidence:.2%}")
            
            # Additional analysis
            st.write("### Analysis Details")
            st.write(f"Based on the provided features, the model predicts with "
                    f"{confidence:.2%} confidence that the company {result_text.lower()}.")
else:
    st.error("Failed to load model. Please check if model files exist in the correct location.")

# Add footer
st.markdown("---")
st.markdown("*This is a multiple model bankruptcy prediction system. Please ensure all input "
           "features are accurate for reliable predictions.*")
