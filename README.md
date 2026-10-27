# AI-Driven ECG Diagnostic Assistant

An Explainable Neuro-Symbolic System for Myocardial Infarction Detection. This system classifies ECG images into four clinical categories: Myocardial Infarction (MI), History of MI, Abnormal Heartbeat, and Normal.

Model Performance:
The retrained Stacking Ensemble achieved a 98.9% accuracy on the test set.

## Technical Architecture

1. **Vision Engine:** Utilizes a pre-trained **MobileNetV2** (PyTorch) to convert raw ECG images into 1,280-dimensional feature vectors.
2. **Preprocessing:** Implements **Otsu’s Binarization** (OpenCV) to remove background grid noise and isolate the ECG signal.
3. **Search Heuristics:** Compares Genetic Algorithms (GA) vs. Sequential Forward Selection (SFS) for feature selection.
4. **Ensemble Tournament:** Uses a **Stacking Ensemble** (Random Forest, SVM, Gradient Boosting + Logistic Regression meta-learner) for high-precision classification.
5. **Explainable AI (XAI):** Integrates **LIME** to generate visual and textual explanations for the model's predictions.

## Dataset
The model was trained on the `heart_disease_prediction_image` dataset from Kaggle, containing 3,972 ECG images across the four classes.

## Project Structure
api_server.py: Flask REST API for model inference.

main_pipeline.py: End-to-end training and evaluation pipeline.

prepare_data.py: Script to format the Kaggle dataset into the required folder structure.

src/: Core backend modules (ingestion, features, ensemble, reasoning).

ui/: Streamlit web application.
