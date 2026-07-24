import streamlit as st
import requests


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="OpenHousing",
    page_icon="🏠",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------

st.title("🏠 OpenHousing Price Prediction")

st.write(
    "Predict the median value of a house using socio-economic "
    "and housing indicators."
)


# -----------------------------
# API Configuration
# -----------------------------

API_URL = "http://api:8000/predict"


# -----------------------------
# Input Form
# -----------------------------

st.header("Enter Housing Information")


col1, col2 = st.columns(2)


with col1:

    crim = st.number_input(
        "Crime Rate (CRIM)",
        value=0.00632
    )

    zn = st.number_input(
        "Residential Land Zone (ZN)",
        value=18.0
    )

    indus = st.number_input(
        "Industrial Area (INDUS)",
        value=2.31
    )

    chas = st.number_input(
        "Charles River (CHAS)",
        value=0.0
    )

    nox = st.number_input(
        "Nitric Oxide Concentration (NOX)",
        value=0.538
    )

    rm = st.number_input(
        "Average Number of Rooms (RM)",
        value=6.575
    )

    age = st.number_input(
        "Age of Houses (AGE)",
        value=65.2
    )


with col2:

    dis = st.number_input(
        "Distance to Employment Centers (DIS)",
        value=4.09
    )

    rad = st.number_input(
        "Highway Accessibility (RAD)",
        value=1.0
    )

    tax = st.number_input(
        "Property Tax Rate (TAX)",
        value=296.0
    )

    ptratio = st.number_input(
        "Pupil-Teacher Ratio (PTRATIO)",
        value=15.3
    )

    b = st.number_input(
        "Black Population Index (B)",
        value=396.90
    )

    lstat = st.number_input(
        "Lower Socio-Economic Status (LSTAT)",
        value=4.98
    )


# -----------------------------
# Prediction Button
# -----------------------------

if st.button(
    "Predict House Price",
    type="primary"
):

    payload = {

        "crim": crim,

        "zn": zn,

        "indus": indus,

        "chas": chas,

        "nox": nox,

        "rm": rm,

        "age": age,

        "dis": dis,

        "rad": rad,

        "tax": tax,

        "ptratio": ptratio,

        "b": b,

        "lstat": lstat

    }


    try:

        response = requests.post(

            API_URL,

            json=payload

        )


        if response.status_code == 200:

            result = response.json()


            st.success(
                "Prediction generated successfully!"
            )


            st.metric(

                label="Predicted Median House Value",

                value=f"{result['predicted_medv']:.2f}"

            )


            st.info(

                f"Model Version: "
                f"{result['model_version']}"

            )


        else:

            st.error(

                f"API Error: "
                f"{response.status_code}"

            )


    except requests.exceptions.ConnectionError:

        st.error(

            "Could not connect to FastAPI. "
            "Please make sure the API is running."

        )