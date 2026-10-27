import cv2
import joblib
import torch
import numpy as np
import streamlit as st
from PIL import Image
import sys
import os

# THIS MUST BE AT THE VERY TOP
sys.path.append(r"C:\ecg-classifier")

# fmt: off
from src.ingestion import preprocess_ecg_image
from src.features import ECGFeatureExtractor
from src.reasoning import ExplanationEngine
# fmt: on


st.set_page_config(page_title="AI ECG Diagnostic Tool", layout="centered")


@st.cache_resource
def load_resources():
    """Loads the model and feature extractor once to keep the app fast."""
    extractor = ECGFeatureExtractor()
    ensemble = joblib.load("models/v1.0/final_ensemble.pkl")
    mask = joblib.load("models/v1.0/ga_feature_mask.pkl")
    classes = [
        "Myocardial Infarction (MI)",
        "History of MI",
        "Abnormal Heartbeat",
        "Normal"
    ]
    return extractor, ensemble, mask, classes


st.title("🫀 AI-Powered ECG Classifier")
st.write("Upload an ECG image to detect Myocardial Infarction or abnormalities.")

# Load models
try:
    extractor, ensemble, mask, class_names = load_resources()

    uploaded_file = st.file_uploader(
        "Choose an ECG Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. Display the original image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded ECG', use_column_width=True)

        with st.spinner('Analyzing...'):
            # Save temp file for OpenCV
            temp_path = "temp_ecg.png"
            image.save(temp_path)

            # 2. Preprocess
            processed_img = preprocess_ecg_image(temp_path)

            # 3. Extract Features
            features = extractor.extract(np.array([processed_img]))

            # 4. Apply Genetic Mask
            selected_features = np.where(mask == 1)[0]
            optimized_features = features[:, selected_features]

            # 5. Predict
            prediction = ensemble.predict(optimized_features)[0]
            probabilities = ensemble.predict_proba(optimized_features)[0]

            # --- NEW: 6. Reasoning Layer ---
            explainer = ExplanationEngine(class_names)
            explanation_text, action_text = explainer.generate_explanation(
                prediction, probabilities)

            # 7. Results
            st.subheader(f"Diagnosis: **{class_names[prediction]}**")

            # Show confidence bars
            st.write("### Confidence Levels:")
            for i, prob in enumerate(probabilities):
                st.write(f"{class_names[i]}")
                st.progress(float(prob))

            # --- NEW: Display the Logic ---
            st.write("---")
            st.subheader("🧠 AI Reasoning & Logic")
            st.info(explanation_text)
            if action_text:
                st.warning(action_text)

except FileNotFoundError:
    st.error(
        "Model files not found! Please run main_pipeline.py first to train and save the model.")
