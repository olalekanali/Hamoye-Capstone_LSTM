import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load model
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

model = load_lstm_model()

# UI
st.title("Predictive Maintenance: RUL Prediction")
st.write("Enter sensor readings to predict Remaining Useful Life (RUL).")

# Input form
sensor1 = st.number_input("Sensor 1 reading", value=0.0)
sensor2 = st.number_input("Sensor 2 reading", value=0.0)
sensor3 = st.number_input("Sensor 3 reading", value=0.0)
# Add more sensors based on your model's input features

if st.button("Predict RUL"):
    try:
        # Prepare input (adjust shape as needed for your model)
        input_data = np.array([[sensor1, sensor2, sensor3]])  # shape (1, timesteps, features)
        input_data = input_data.reshape((1, input_data.shape[1], 1))  # Modify based on model's expected input

        # Predict
        prediction = model.predict(input_data)
        st.success(f"Predicted RUL: {prediction[0][0]:.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
