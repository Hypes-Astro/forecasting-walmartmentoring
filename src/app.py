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
    header.title("Walmart Sales Prediction by Adnya")
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.write(' ')

    # with col2:
    #     st.image(image="../banner.jpeg", caption="Walmart", width=300)

    # with col3:
    #     st.write(' ')

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

         # Button to view the chart
        st.write("Graph showing daily sales can be viewed below")
        if st.button("View Chart"):

            data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

            # Sort the data by Date
            data = data.sort_values(by='Date')
            # Set the "Date" column as the index
            load_df = data.set_index('Date')

            # Display the line chart with dates on the x-axis
            st.subheader("A Chart of the Daily Sales Across Favorita Stores")
            st.line_chart(load_df["Weekly_Sales"])

    