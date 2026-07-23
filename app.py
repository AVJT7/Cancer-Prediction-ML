import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("svm_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Cancer Prediction", page_icon="🩺")

st.title("🩺 Breast Cancer Prediction using Machine Learning")

st.write("Enter the patient details below and click Predict.")
st.header("Enter Feature Values")

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)
    # Convert input into DataFrame
input_data = pd.DataFrame([inputs], columns=feature_names)

# Scale the input
scaled_data = scaler.transform(input_data)

# Prediction Button
if st.button("Predict"):

    prediction = model.predict(scaled_data)

    if prediction[0] == 'B':
        st.success("Prediction: Benign (No Cancer Detected)")
    else:
        st.error("Prediction: Malignant (Cancer Detected)")