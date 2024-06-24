

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model, fitted scalar and the list of required columns:
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
required_columns = joblib.load('required_columns.pkl')

# Define the input fields
def user_input_features():
    patient_age = st.number_input('Age', min_value=0, max_value=120, value=25)
    patient_sex = st.selectbox('Sex', ('male', 'female'))
    chest_pain_type = st.selectbox('Chest Pain Type', 
                                   ('typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'))
    resting_blood_pressure = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=50, max_value=200, value=120)
    serum_cholesterol = st.number_input('Serum Cholesterol (in mg/dl)', min_value=100, max_value=400, value=200)
    fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('false', 'true'))
    resting_ecg = st.selectbox('Resting ECG', ('normal', 'abnormal', 'ventricular hypertrophy'))
    max_heart_rate = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, value=120)
    exercise_induced_angina = st.selectbox('Exercise Induced Angina', ('no', 'yes'))
    st_depression = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=6.0, value=0.0)
    st_slope = st.selectbox('Slope of Peak Exercise ST Segment', ('upsloping', 'flat', 'downsloping'))
    nr_vessels_colored = st.selectbox('Number of Major Vessels Colored by Flourosopy', 
                                      ('zero', 'one', 'two', 'three', 'four'))
    heart_status = st.selectbox('Heart Status', ('unknown', 'normal', 'fixed defect', 'reversible defect'))

    data = {
        'patient_age': patient_age,
        'patient_sex': patient_sex,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'serum_cholesterol': serum_cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'resting_ecg': resting_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression': st_depression,
        'st_slope': st_slope,
        'nr_vessels_colored': nr_vessels_colored,
        'heart_status': heart_status
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Convert user input to model input
def preprocess_input(df):
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Add missing columns with default value of 0
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[required_columns]
    
    # Scale numerical features
    df[['patient_age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate', 'st_depression']] = scaler.transform(
        df[['patient_age', 'resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate', 'st_depression']])
    
    return df

# Main application
st.title('Heart Disease Prediction App')
st.write("Enter the patient's details to predict the likelihood of heart disease")

input_df = user_input_features()
st.write(input_df)

processed_input = preprocess_input(input_df)

# Predict
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

st.subheader('Prediction')
heart_disease_status = 'Heart Disease' if prediction[0] == 1 else 'Normal'
st.write(f"The patient is likely to have: **{heart_disease_status}**")

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Error handling and documentation
try:
    pass
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
