import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_Diabetes(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title("Diabetes Survival Prediction")

# Input fields for each parameter
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=14, value=0, step=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=180, value=0, step=1)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=125, value=0, step=1)
SkinThickness = st.number_input("SkinThickness", min_value=0, max_value=120, value=0, step=1)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0, step=1)
BMI = st.number_input("BMI", min_value=0.0, max_value=68.9, value=0.0, step=0.1)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.078, max_value=3.12, value=0.1, step=0.1)
Age = st.number_input("Age", min_value=5, max_value=100, value=20, step=1)


# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_Diabetes(input_data)

        if pred == 1:
            # Diabetes
            st.error(f"Prediction: Diabetes with probability {prob:.2f}")
        else:
            # No Diabetes
            st.success(f"Prediction: No Diabetes {prob:.2f}")
