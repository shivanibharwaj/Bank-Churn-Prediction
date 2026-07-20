import streamlit as st
import pickle
import numpy as np


# Load model and scaler

model = pickle.load(open("models/customer_churn_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))


st.title("Customer Bank Churn Prediction")

st.write(
    "Predict whether a customer is likely to leave the bank."
)


credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=600
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    value=50000.0
)

num_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_card = st.selectbox(
    "Has Credit Card",
    [0,1]
)

active_member = st.selectbox(
    "Active Member",
    [0,1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    value=50000.0
)


gender = st.selectbox(
    "Gender",
    ["Female","Male"]
)


geography = st.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)


if st.button("Predict"):

    gender_value = 1 if gender=="Male" else 0

    geo_germany = 1 if geography=="Germany" else 0
    geo_spain = 1 if geography=="Spain" else 0


    input_data = np.array([[
        credit_score,
        gender_value,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        active_member,
        estimated_salary,
        geo_germany,
        geo_spain
    ]])


    input_scaled = scaler.transform(input_data)


    prediction = model.predict(input_scaled)


    if prediction[0] == 1:
        st.error(
            "Customer is likely to churn"
        )
    else:
        st.success(
            "Customer is likely to stay"
        )
