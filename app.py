# streamlit_app.py
import streamlit as st
import os
import json
from OOP_Predictor import HotelBookingPredictor

# Initialize predictor
model_path = os.path.join("models", "best_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")
encoder_path = os.path.join("models", "encoder.pkl")
predictor = HotelBookingPredictor(model_path, scaler_path, encoder_path)

# Manual JSON input
st.subheader("📥 Load Booking Data from Object")

if "load_triggered" not in st.session_state:
    st.session_state.load_triggered = False
if "manual_input_text" not in st.session_state:
    st.session_state.manual_input_text = ""

if st.session_state.load_triggered:
    st.session_state.manual_input_text = ""
    st.session_state.load_triggered = False

manual_json = st.text_area(
    "Paste booking data here (JSON format):",
    value=st.session_state.manual_input_text,
    key="manual_input_text"
)

if st.button("📤 Load Input into Form"):
    try:
        parsed = json.loads(st.session_state.manual_input_text)
        for key, value in parsed.items():
            st.session_state[key] = value
        st.session_state.load_triggered = True  # triggers clear on next render
        st.rerun()  # force rerun so clearing happens cleanly
    except Exception as e:
        st.error(f"⚠️ Invalid JSON: {e}")

st.title("🏨 Hotel Booking/Cancellation Prediction")
st.subheader("Predict booking/cancellations with machine learning")
st.divider()

# Input form in expandable sections
with st.expander("Guest Information 🧑👧👦", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        no_of_adults = st.number_input("Adults", 0, 4, 2, key="no_of_adults")
        no_of_children = st.number_input("Children", 0, 10, 0, key="no_of_children")
        meal_plan = st.radio("Meal Plan", ["Meal Plan 1", "Not Selected", "Meal Plan 2"],
                             horizontal=True, index=1, key="type_of_meal_plan")

    with col2:
        parking = st.radio("Parking Required", ["No", "Yes"], horizontal=True, index=0, key="required_car_parking_space")
        room_type = st.selectbox("Room Type", [f"Room_Type {i}" for i in range(1, 8)], index=0, key="room_type_reserved")
        special_requests = st.slider("Special Requests", 0, 5, 0, key="no_of_special_requests")

with st.expander("Booking Details 📅", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        lead_time = st.slider("Lead Time (days)", 0, 500, 224, key="lead_time")
        arrival_date = st.number_input("Arrival Date", 1, 31, 15, key="arrival_date")
        arrival_month = st.select_slider("Arrival Month", options=range(1, 13), value=6, key="arrival_month")

    with col2:
        arrival_year = st.radio("Arrival Year", [2017, 2018], horizontal=True, index=1, key="arrival_year")
        weekend_nights, week_nights = st.slider(
            "Nights Distribution",
            min_value=0, max_value=20,
            value=(1, 5), step=1,
            format="%d nights"
        )

with st.expander("Customer History 📊", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        market_segment = st.select_slider(
            "Market Segment",
            options=["Aviation", "Complementary", "Corporate", "Offline", "Online"],
            value="Online",
            key="market_segment_type"
        )
        prev_canc = st.number_input("Previous Cancellations", 0, 13, 0, key="no_of_previous_cancellations")

    with col2:
        prev_success = st.number_input("Successful Bookings", 0, 58, 0, key="no_of_previous_bookings_not_canceled")
        repeated_guest = st.checkbox("Repeated Guest", value=False, key="repeated_guest")

with st.expander("Financial Information 💰", expanded=True):
    avg_price = st.slider("Average Price per Room", 0.0, 540.0, 100.0, step=0.5, key="avg_price_per_room")
    total_cost = avg_price * (weekend_nights + week_nights)
    st.metric(label="Total Estimated Cost", value=f"${total_cost:,.2f}")

# Prediction button
st.divider()
predict_button = st.button("Predict Cancellation 🔍", use_container_width=True)

if predict_button:
    input_data = {
        "no_of_adults": st.session_state.no_of_adults,
        "no_of_children": st.session_state.no_of_children,
        "no_of_weekend_nights": weekend_nights,
        "no_of_week_nights": week_nights,
        "type_of_meal_plan": st.session_state.type_of_meal_plan,
        "required_car_parking_space": 1 if st.session_state.required_car_parking_space == "Yes" else 0,
        "room_type_reserved": st.session_state.room_type_reserved,
        "lead_time": st.session_state.lead_time,
        "arrival_year": st.session_state.arrival_year,
        "arrival_month": st.session_state.arrival_month,
        "arrival_date": st.session_state.arrival_date,
        "market_segment_type": st.session_state.market_segment_type,
        "repeated_guest": 1 if st.session_state.repeated_guest else 0,
        "no_of_previous_cancellations": st.session_state.no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": st.session_state.no_of_previous_bookings_not_canceled,
        "avg_price_per_room": st.session_state.avg_price_per_room,
        "no_of_special_requests": st.session_state.no_of_special_requests
    }

    processed_input = predictor.preprocess_input(input_data)
    prediction = predictor.model.predict(processed_input)[0]
    prediction_proba = predictor.model.predict_proba(processed_input)[0]

    if prediction == 1:
        st.success(f"✔️ Booking Likely to be Completed ({prediction_proba[1]*100:.1f}% confidence)")
        st.balloons()
    else:
        st.error(f"❌ High Risk of Cancellation ({prediction_proba[0]*100:.1f}% confidence)")
        st.snow()
