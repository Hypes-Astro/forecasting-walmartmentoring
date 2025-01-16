import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# Sidebar information
st.sidebar.title("Environment Info")
st.sidebar.write(f"TensorFlow Version: {tf.__version__}")

def load_model(path):
    try:
        st.sidebar.write(f"Loading model from: {path}")
        model = tf.keras.models.load_model(path, compile=False)
        st.sidebar.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None

def load_data(path):
    if os.path.exists(path):
        data = pd.read_csv(path)
        st.sidebar.success("Data loaded successfully!")
        return data
    else:
        st.sidebar.error(f"Dataset file not found at path: {path}")
        return None

def prepare_input_sequence(month_num, sequence_length=25):
    """
    Prepare input sequence with the correct shape for the model.
    Args:
        month_num: The selected month number (1-12)
        sequence_length: The expected sequence length for the model
    Returns:
        Properly shaped input array
    """
    # Create a sequence leading up to the selected month
    sequence = np.zeros(sequence_length)
    sequence[-1] = month_num  # Put the month number at the last position
    
    # Reshape to match model's expected input shape: (batch_size, timesteps, features)
    return np.array([sequence]).reshape(1, 1, sequence_length)

# File paths
data_path = "data/Walmart.csv"
model_path = "model/best_model.h5"

# Load data and model
data = load_data(data_path)
model = load_model(model_path)

# Header Application
st.title("Monthly Sales Prediction")
st.write("This app predicts sales based on the selected month.")

# Months list
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Month to number mapping
month_mapping = {month: i + 1 for i, month in enumerate(months)}

# Month Input
st.header("Make a Prediction")
selected_month = st.selectbox("Select a month", months)

# Prediction Button
if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Please check the model path.")
    else:
        try:
            # Debug information
            st.write("Preparing input data...")
            month_num = month_mapping[selected_month]
            
            # Prepare input with correct shape
            input_data = prepare_input_sequence(month_num)
            st.write(f"Input data shape: {input_data.shape}")

            # Make prediction
            st.write("Making prediction...")
            prediction = model.predict(input_data, verbose=0)
            st.write(f"Raw prediction: {prediction}")

            # Display results
            rounded_prediction = np.round(prediction[0][0], 2)
            st.success(f"The predicted sales for {selected_month} is **${rounded_prediction:,.2f}**")
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Detailed error information:", str(e))