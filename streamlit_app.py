import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

#Load model
with open('model/bestmodel7003.pkl','rb') as file:
    model = pickle.load(file)

# Load image for title
image = Image.open('image.png')

# Input features
def user_input_features():

    age = st.sidebar.number_input('Age', min_value=20, max_value=80, value=40)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
    resting_bp_s = st.sidebar.number_input('Resting Blood Pressure (mmHg)', min_value=60, max_value=200, value=120)
    cholesterol = st.sidebar.number_input('Cholesterol (mg/dl)', min_value=60, max_value=600, value=200)
    fasting_blood_sugar = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
    resting_ecg = st.sidebar.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality','Left ventricular hypertrophy'])
    max_heart_rate = st.sidebar.number_input('Maximum Heart Rate', min_value=60, max_value=220, value=150)
    exercise_angina = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = st.sidebar.number_input('ST Depression (Oldpeak)', min_value=0.0, max_value=6.0, value=0.0)
    st_slope = st.sidebar.selectbox('ST Slope', ['Unsloping', 'Flat', 'Downsloping'])

    # Combine the features into a dataframe and Convert features to binary
    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'chest pain type': 1 if chest_pain_type == 'Typical angina' else 2 if chest_pain_type == 'Atypical angina' else 3 if chest_pain_type == 'Non-anginal pain' else 4,
        'resting bp s': resting_bp_s,
        'cholesterol': cholesterol,
        'fasting blood sugar': 1 if fasting_blood_sugar == 'True' else 0,
        'resting ecg': 0 if resting_ecg == 'Normal' else 1 if resting_ecg == 'ST-T wave abnormality' else 2,
        'max heart rate': max_heart_rate,
        'exercise angina': 1 if exercise_angina == 'Yes' else 0,
        'oldpeak': oldpeak,
        'ST slope': 1 if st_slope == 'Unsloping' else 2 if st_slope == 'Flat' else 3
    }

    features = pd.DataFrame(data, index=[0])
    return features

# App title and description
col1, col2 = st.columns([1, 2])
with col1:
   st.image(image, width=150)
   
with col2:
   st.title("Heart Disease Prediction Application")

# Custom Styling for description
st.markdown("""
    <style>
    .description {
        border: 1px solid #4F8BF9;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        color: #FFFFFF; /* White font color */
        background-color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# Introduction text with border
st.markdown("""
    <div class="description">
        <p>Welcome to the Heart Disease Prediction Application. 
        This tool is designed to help you understand the likelihood of developing heart disease in individuals using medical attributes. 
        Simply adjust the parameters in the sidebar to match your details and click 'Predict' to see the outcome.</p>
    </div>
    """, unsafe_allow_html=True)

predict_button = st.button('Predict')

# Sidebar for user input features
with st.sidebar:
    st.header("User Input Features")
    input_df = user_input_features()

# Main prediction logic
custom_threshold = 0.7

if predict_button:
    # Get probability of the positive class
    probability = model.predict_proba(input_df)[0][1] * 100

    # Apply custom threshold to determine class
    prediction = 1 if (probability / 100) > custom_threshold else 0

    # Convert prediction to interpretable output
    risk_level = "HIGH" if prediction == 1 else "LOW"
    probability_text = f"{probability:.2f}%"

    # Display the prediction with styling
    st.subheader("Prediction Result ")

    # Use color to emphasize the risk level
    if risk_level == "HIGH":
        risk_color = "red"
    else:
        risk_color = "green"

    st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
    st.write(f"The model predicts a **{risk_level}** risk of developing heart disease, "f"with a probability of **{probability_text}**.")

    # Display recommendations based on risk level
    st.subheader("Recommendations")
    if risk_level == "HIGH":
       st.error("Based on the high risk assessment, we recommend:")
       st.write("• Schedule an appointment with a cardiologist for detailed evaluation")
       st.write("• Monitor blood pressure and cholesterol levels regularly")
       st.write("• Consider lifestyle modifications including diet and exercise")

    else:
       st.success("Based on the low risk assessment, we recommend:")
       st.write("• Maintain regular health check-ups")
       st.write("• Continue healthy lifestyle habits")
       st.write("• Stay physically active")
       

