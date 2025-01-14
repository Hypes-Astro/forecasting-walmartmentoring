import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Define functions
def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

# Load dataset
# Load the dataset
data_path = "data/Walmart.csv"
data = load_data(data_path)

st.set_page_config(
    page_title="Forecasting Walmart - Adnya",
    page_icon="✨"
)

#define app section
header=st.container()
prediction=st.container()
konten = st.container()

#define header
with header:
    header.title("Retails Sales Demand Forecasting")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image(image="../banner.jpeg", caption="Walmart", width=300)

    with col3:
        st.write(' ')

    header.write("On this page,you can predict sales")

# Create lists
inputs = ["date"]


with konten:
    # konten
    with st.expander("Make a prediction", expanded=True):
        # input
        date = st.date_input(label="Enter a date")
        # Create a button
        predicted = st.button("Predict")

        

    