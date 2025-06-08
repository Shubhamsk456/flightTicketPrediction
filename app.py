
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and columns
model = pickle.load(open("flight_price_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))
df = pd.read_excel("Data_Train.xlsx")

st.title("✈️ Flight Price Prediction App")

# Sidebar for navigation
section = st.sidebar.radio("Select Section", ["Dataset", "Graphs", "Prediction"])

if section == "Dataset":
    st.header("Dataset Preview")
    st.write(df.head(50))

elif section == "Graphs":
    st.header("Data Visualization")
    st.subheader("Price vs Airline")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x="Airline", y="Price", data=df, ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("Price vs Total Stops")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Total_Stops", y="Price", data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Price Distribution")
    fig3, ax3 = plt.subplots()
    sns.histplot(df["Price"], kde=True, ax=ax3)
    st.pyplot(fig3)

elif section == "Prediction":
    st.header("Make a Price Prediction")

    airline = st.selectbox("Airline", ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet'])
    source = st.selectbox("Source", ['Delhi', 'Kolkata', 'Mumbai', 'Chennai'])
    destination = st.selectbox("Destination", ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad'])
    total_stops = st.selectbox("Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops'])

    journey_day = st.slider("Journey Day", 1, 31, 15)
    journey_month = st.slider("Journey Month", 1, 12, 6)

    dep_hour = st.slider("Departure Hour", 0, 23, 10)
    dep_min = st.slider("Departure Minute", 0, 59, 0)
    arrival_hour = st.slider("Arrival Hour", 0, 23, 12)
    arrival_min = st.slider("Arrival Minute", 0, 59, 0)

    duration_hours = st.slider("Duration Hours", 0, 24, 2)
    duration_mins = st.slider("Duration Minutes", 0, 59, 0)

    input_data = {
        'Journey_day': journey_day,
        'Journey_month': journey_month,
        'Dep_hour': dep_hour,
        'Dep_min': dep_min,
        'Arrival_hour': arrival_hour,
        'Arrival_min': arrival_min,
        'Duration_hours': duration_hours,
        'Duration_mins': duration_mins,
        'Total_Stops': int(total_stops.split()[0]) if total_stops != 'non-stop' else 0,
    }

    airlines = [f'Airline_{airline}']
    sources = [f'Source_{source}']
    destinations = [f'Destination_{destination}']

    for col in model_columns:
        input_data[col] = 0

    for key in input_data:
        if key in model_columns:
            input_data[key] = input_data[key]

    for cat in airlines + sources + destinations:
        if cat in model_columns:
            input_data[cat] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_columns]

    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Flight Price: ₹ {int(prediction)}")
