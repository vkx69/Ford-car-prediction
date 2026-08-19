import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Exact features used during training
expected_columns = list(model.feature_names_in_)

# =====================================================
# FORD MODELS
# =====================================================

ford_models = [
    "C-MAX",
    "EcoSport",
    "Edge",
    "Escort",
    "Fiesta",
    "Focus",
    "Fusion",
    "Galaxy",
    "Grand C-MAX",
    "Grand Tourneo Connect",
    "KA",
    "Ka+",
    "Kuga",
    "Mondeo",
    "Mustang",
    "Puma",
    "Ranger",
    "S-MAX",
    "Streetka",
    "Tourneo Connect",
    "Tourneo Custom",
    "Transit Tourneo"
]

# =====================================================
# ENCODING
# =====================================================

transmission_map = {
    "Manual": 0,
    "Automatic": 1,
    "Semi-Auto": 2
}

fuel_map = {
    "Petrol": 0,
    "Diesel": 1,
    "Hybrid": 2,
    "Electric": 3,
    "Other": 4
}

# =====================================================
# UI
# =====================================================

st.title("🚗 Ford Car Price Predictor")

st.markdown(
    "### Predict the estimated price of a Ford car using Machine Learning"
)

st.divider()

col1, col2 = st.columns(2)

# =====================================================
# INPUTS
# =====================================================

with col1:

    year = st.number_input(
        "📅 Year",
        min_value=2000,
        max_value=2026,
        value=2019,
        step=1
    )

    car_model = st.selectbox(
        "🚘 Ford Model",
        ford_models
    )

    mileage = st.number_input(
        "🛣️ Mileage",
        min_value=0,
        value=30000,
        step=1000
    )

    transmission_name = st.selectbox(
        "⚙️ Transmission",
        list(transmission_map.keys())
    )

with col2:

    fuel_name = st.selectbox(
        "⛽ Fuel Type",
        list(fuel_map.keys())
    )

    tax = st.number_input(
        "💷 Tax",
        min_value=0,
        value=150,
        step=10
    )

    mpg = st.number_input(
        "⛽ MPG",
        min_value=0.0,
        value=45.0,
        step=0.1
    )

    engine_size = st.number_input(
        "🔧 Engine Size",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1
    )

st.divider()

# =====================================================
# PREDICT
# =====================================================

if st.button(
    "🔮 Predict Car Price",
    use_container_width=True
):

    # Convert categorical values to numeric
    transmission = transmission_map[transmission_name]
    fuel_type = fuel_map[fuel_name]

    # -------------------------------------------------
    # BASIC FEATURES
    # -------------------------------------------------

    input_data = pd.DataFrame({
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    # -------------------------------------------------
    # MODEL ONE-HOT FEATURES
    # -------------------------------------------------

    for column in expected_columns:

        if column.startswith("model_"):
            input_data[column] = 0

    # Selected Ford model
    model_column = "model_" + car_model

    if model_column in input_data.columns:
        input_data[model_column] = 1

    # -------------------------------------------------
    # ADD MISSING FEATURES
    # -------------------------------------------------

    for column in expected_columns:

        if column not in input_data.columns:
            input_data[column] = 0

    # -------------------------------------------------
    # EXACT FEATURE ORDER
    # -------------------------------------------------

    input_data = input_data[expected_columns]

    # Ensure numeric
    input_data = input_data.astype(float)

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        # ------------------------------------------------
        # IF PRICE WAS LOG TRANSFORMED
        # ------------------------------------------------
        #
        # If you trained using:
        #
        # y = np.log1p(df['price'])
        #
        # then use:
        #
        # prediction = np.expm1(prediction)
        #
        # Otherwise leave it commented.

        st.success("✅ Prediction Successful!")

        st.metric(
            label="💰 Estimated Car Price",
            value=f"£{prediction:,.2f}"
        )

        st.divider()

        st.subheader(f"🚘 Ford {car_model}")

        info1, info2, info3, info4 = st.columns(4)

        with info1:
            st.write(f"**Year**")
            st.write(year)

        with info2:
            st.write(f"**Mileage**")
            st.write(f"{mileage:,} miles")

        with info3:
            st.write(f"**Transmission**")
            st.write(transmission_name)

        with info4:
            st.write(f"**Fuel**")
            st.write(fuel_name)

    except Exception as e:

        st.error(f"Prediction Error: {e}")

        st.write("Data sent to model:")
        st.dataframe(input_data)