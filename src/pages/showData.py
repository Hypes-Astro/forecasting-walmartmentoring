import streamlit as st
import pandas as pd
import plotly
import plotly.express as px 

# Define functions
def load_data(path):
    dataset = pd.read_csv(path)
    return dataset

# Load dataset
# Load the dataset
data_path = "data/Walmart.csv"
data = load_data(data_path)


konten = st.container()

with konten:
    konten.title("Data preview")
    konten.write(data)

    st.write("View the model's prediction")
    if st.button("Model's graph"):

        # we predict with the model
        # result = model.predict(test_df)

        # Create a Plotly line chart for the model's predictions
        # fig = px.line(x=test_df.index, y=result, title="Model's Forecast")

        # Display the chart using st.plotly_chart()
        st.subheader("A plot of model's forecast")
        st.plotly_chart(fig)
