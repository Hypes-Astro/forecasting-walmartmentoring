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
    # if st.button("Model's graph"):

        # we predict with the model
        # result = model.predict(test_data)

        # Create a Plotly line chart for the model's predictions
        # fig = px.line(x=test_data.index, y=result, title="Model's Forecast")

        # Display the chart using st.plotly_chart()
        # st.subheader("A plot of model's forecast")
        # st.plotly_chart(fig)

    st.write("View grafik")
    # Button to view the chart
    st.write("Graph showing daily sales can be viewed below")
    

    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

       # Tambahkan widget selectbox untuk memilih store
    store_options = data['Store'].unique()
    selected_store = st.selectbox("Pilih Store", store_options)

        # Filter data berdasarkan store yang dipilih
    filtered_data = data[data['Store'] == selected_store]

        # Urutkan data berdasarkan tanggal
    filtered_data = filtered_data.sort_values(by='Date')

        # Set kolom "Date" sebagai index
    load_data = filtered_data.set_index('Date')

        # Tampilkan chart
    st.subheader(f"A Chart of the Daily Sales for Store {selected_store}")
    st.line_chart(load_data["Weekly_Sales"])
