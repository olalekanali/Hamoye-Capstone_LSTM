import os
import streamlit as st
import numpy as np
from keras.models import load_model

# Load model
@st.cache_resource
def load_lstm_model():
    model_path = os.path.join(os.path.dirname(__file__), "lstm_rul_model.h5")
    return load_model(model_path, compile=False)

model = load_lstm_model()

# UI
st.title("Predictive Maintenance: RUL Prediction")
st.markdown("Enter **14 sensor readings** to predict the Remaining Useful Life (RUL) of a machine.")

# Input: 14 sensor readings
st.subheader("Sensor Readings")
sensor_inputs = []

cols = st.columns(2)
for i in range(14):
    col = cols[i % 2]  # Alternate between 2 columns
    sensor_value = col.number_input(f"Sensor {i + 1}", value=0.0, step=0.1, format="%.2f")
    sensor_inputs.append(sensor_value)

# Predict button
if st.button("Predict RUL"):
    try:
        # Prepare input
        input_data = np.array(sensor_inputs).reshape(1, 1, 14)  # (1 sample, 1 timestep, 14 features)
        
        # Predict
        prediction = model.predict(input_data)
        st.success(f"Predicted Remaining Useful Life: **{prediction[0][0]:.2f} cycles**")
    
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
