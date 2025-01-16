import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime, timedelta

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

def prepare_sequence(month_num, week_num, lookback=25):
    """
    Prepare input sequence for LSTM model considering weeks.
    Args:
        month_num: Selected month (1-12)
        week_num: Week number in the month (1-4)
        lookback: Sequence length
    Returns:
        Sequence shaped for Bidirectional LSTM
    """
    sequence = []
    current_month = month_num
    
    # Mengisi sequence mundur dari bulan yang dipilih
    for i in range(lookback):
        if current_month < 1:
            current_month = 12
        sequence.insert(0, current_month + (week_num / 4))  # Menambahkan fraksi minggu
        current_month -= 1
    
    # Normalisasi sequence
    sequence = np.array(sequence) / 12.0
    
    return np.array(sequence).reshape(1, 1, lookback)

def get_week_dates(month, year=2024):
    """
    Mendapatkan tanggal untuk setiap minggu dalam bulan tertentu.
    """
    first_day = datetime(year, month, 1)
    weeks = []
    
    for week in range(4):
        week_start = first_day + timedelta(days=week*7)
        week_end = week_start + timedelta(days=6)
        if week_end.month != month:
            week_end = datetime(year, month + 1 if month < 12 else 1, 1) - timedelta(days=1)
        weeks.append(f"{week_start.strftime('%d')} - {week_end.strftime('%d %B')}")
    
    return weeks

# File paths
data_path = "data/Walmart.csv"
model_path = "model/best_model.h5"

# Load data and model
data = load_data(data_path)
model = load_model(model_path)

# Header Application
st.title("Monthly Sales Prediction (Weekly Breakdown)")
st.write("This app predicts sales for each week of the selected month.")

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
            month_num = month_mapping[selected_month]
            week_dates = get_week_dates(month_num)
            
            # Create a container for weekly predictions
            st.subheader(f"Weekly Sales Predictions for {selected_month}")
            
            # Metrics container
            col1, col2 = st.columns(2)
            
            weekly_predictions = []
            total_sales = 0
            
            # Predict for each week
            for week in range(4):
                input_data = prepare_sequence(month_num, week + 1)
                prediction = model.predict(input_data, verbose=0)
                weekly_pred = float(prediction[0][0])
                weekly_predictions.append(weekly_pred)
                total_sales += weekly_pred
            
            # Display weekly predictions
            for week, (pred, dates) in enumerate(zip(weekly_predictions, week_dates), 1):
                st.metric(
                    label=f"Week {week} ({dates})",
                    value=f"${pred:,.2f}",
                    delta=f"{((pred/total_sales)*100):.1f}% of monthly total"
                )
            
            # Display total
            st.markdown("---")
            st.metric(
                label="Projected Monthly Total",
                value=f"${total_sales:,.2f}",
                delta=f"Average: ${(total_sales/4):,.2f} per week"
            )
            
            # Visualisasi tambahan
            if st.checkbox("Show Details"):
                st.write("Weekly Breakdown:")
                df_weekly = pd.DataFrame({
                    'Week': [f"Week {i+1}" for i in range(4)],
                    'Dates': week_dates,
                    'Sales': weekly_predictions,
                    'Percentage': [(pred/total_sales)*100 for pred in weekly_predictions]
                })
                st.dataframe(df_weekly.style.format({
                    'Sales': '${:,.2f}',
                    'Percentage': '{:.1f}%'
                }))
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Detailed error information:", str(e))