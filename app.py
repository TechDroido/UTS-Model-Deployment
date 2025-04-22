# streamlit_app.py
import streamlit as st
import streamlit.components.v1 as components
from OOP_Predictor import HotelBookingPredictor
import os
import json

# Initialize predictor
model_path = os.path.join("models", "best_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")
encoder_path = os.path.join("models", "encoder.pkl")
predictor = HotelBookingPredictor(model_path, scaler_path, encoder_path)

# Page setup
st.set_page_config(
    page_title="🏨 Hotel Booking Cancellation Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Test Cases ---
json_test_case = """{
  "type_of_meal_plan": "Meal Plan 2",
  "room_type_reserved": "Room_Type 4",
  "market_segment_type": "Offline",
  "lead_time": 3,
  "avg_price_per_room": 180.0,
  "no_of_adults": 1,
  "no_of_children": 0,
  "no_of_weekend_nights": 2,
  "no_of_week_nights": 2,
  "no_of_previous_bookings_not_canceled": 1,
  "no_of_previous_cancellations": 0,
  "no_of_special_requests": 2
}"""

test_case_1 = {
    "type_of_meal_plan": "Meal Plan 1",
    "room_type_reserved": "Room_Type 1",
    "market_segment_type": "Online",
    "lead_time": 5,
    "avg_price_per_room": 75.0,
    "no_of_adults": 2,
    "no_of_children": 1,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "no_of_previous_bookings_not_canceled": 3,
    "no_of_previous_cancellations": 0,
    "no_of_special_requests": 2
}
test_case_2 = {
    "type_of_meal_plan": "Not Selected",
    "room_type_reserved": "Room_Type 6",
    "market_segment_type": "Offline",
    "lead_time": 320,
    "avg_price_per_room": 420.0,
    "no_of_adults": 1,
    "no_of_children": 0,
    "no_of_weekend_nights": 2,
    "no_of_week_nights": 5,
    "no_of_previous_bookings_not_canceled": 0,
    "no_of_previous_cancellations": 1,
    "no_of_special_requests": 0
}

# --- Header ---
st.title("🏨 Hotel Booking/Cancellation Prediction")
st.subheader("Predict booking/cancellations with machine learning")
st.divider()

# --- JSON Input + Test Case Controls ---
st.markdown("### 🔧 Input Setup")

col_json, col_test = st.columns(2)

with col_json:
    st.markdown("#### 📥 Paste JSON Input")
    json_input = st.text_area("Enter booking data as JSON", value=json_test_case, height=250)
    if st.button("📤 Load JSON into Form"):
        try:
            parsed = json.loads(json_input)
            for key, value in parsed.items():
                st.session_state[key] = value
            st.success("✅ Parameters loaded from JSON.")
        except Exception as e:
            st.error(f"❌ Invalid JSON: {e}")

with col_test:
    st.markdown("#### 🧪 Quick Test Cases")

    col_btn1, col_btn2 = st.columns(2)

    case_loaded = None  # Track which case was clicked

    with col_btn1:
        if st.button("💼 Case 1", use_container_width=True):
            for key, value in test_case_1.items():
                st.session_state[key] = value
            case_loaded = "💼 Case 1"

    with col_btn2:
        if st.button("💼 Case 2", use_container_width=True):
            for key, value in test_case_2.items():
                st.session_state[key] = value
            case_loaded = "💼 Case 2"

    # Display success message dynamically
    if case_loaded:
        custom_css = """
        <style>
        @keyframes fadeIn {
            0%   { opacity: 0; transform: translateY(-10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        .success-banner {
            animation: fadeIn 0.5s ease-out;
            margin-top: 1rem;
            padding: 1rem 1.5rem;
            border-left: 6px solid #2ecc71;
            background-color: rgba(46, 204, 113, 0.15);
            color: #d4fbe6;
            border-radius: 12px;
            font-size: 1.1rem;
            font-family: Inter;
        }
        </style>
        """

        success_html = f"""
        <div class="success-banner">✅ <strong>{case_loaded}</strong> loaded into the form.</div>
        """

        components.html(custom_css + success_html, height=100)

# --- Input Form ---
with st.expander("Guest Information 🧑👧👦", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        no_of_adults = st.number_input("Adults", 0, 4, st.session_state.get("no_of_adults", 2))
        no_of_children = st.number_input("Children", 0, 10, st.session_state.get("no_of_children", 0))
        meal_plan = st.radio("Meal Plan", ["Meal Plan 1", "Not Selected", "Meal Plan 2"], 
                             horizontal=True, index=["Meal Plan 1", "Not Selected", "Meal Plan 2"].index(st.session_state.get("type_of_meal_plan", "Not Selected")))
    with col2:
        parking = st.radio("Parking Required", ["No", "Yes"], horizontal=True, index=0)
        room_type = st.selectbox("Room Type", [f"Room_Type {i}" for i in range(1, 8)], 
                                 index=[f"Room_Type {i}" for i in range(1, 8)].index(st.session_state.get("room_type_reserved", "Room_Type 1")))
        special_requests = st.slider("Special Requests", 0, 5, st.session_state.get("no_of_special_requests", 0))

with st.expander("Booking Details 📅", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        lead_time = st.slider("Lead Time (days)", 0, 500, st.session_state.get("lead_time", 224))
        arrival_date = st.number_input("Arrival Date", 1, 31, 15)
        arrival_month = st.select_slider("Arrival Month", options=range(1,13), value=6)
    with col2:
        arrival_year = st.radio("Arrival Year", [2017, 2018], horizontal=True, index=1)
        weekend_nights = st.slider("Weekend Nights", 0, 20, st.session_state.get("no_of_weekend_nights", 1))
        week_nights = st.slider("Week Nights", 0, 20, st.session_state.get("no_of_week_nights", 5))

with st.expander("Customer History 📊", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        market_segment = st.select_slider("Market Segment",
            options=["Aviation", "Complementary", "Corporate", "Offline", "Online"],
            value=st.session_state.get("market_segment_type", "Online"))
        prev_canc = st.number_input("Previous Cancellations", 0, 13, st.session_state.get("no_of_previous_cancellations", 0))
    with col2:
        prev_success = st.number_input("Successful Bookings", 0, 58, st.session_state.get("no_of_previous_bookings_not_canceled", 0))
        repeated_guest = st.checkbox("Repeated Guest", value=False)

with st.expander("Financial Information 💰", expanded=True):
    avg_price = st.slider("Average Price per Room", 0.0, 540.0, st.session_state.get("avg_price_per_room", 100.0), step=0.5)
    total_cost = avg_price * (weekend_nights + week_nights)
    st.metric(label="Total Estimated Cost", value=f"${total_cost:,.2f}")

# --- Prediction ---
st.divider()
if st.button("🔍 Predict Cancellation", use_container_width=True):
    input_data = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": weekend_nights,
        "no_of_week_nights": week_nights,
        "type_of_meal_plan": meal_plan,
        "required_car_parking_space": 1 if parking == "Yes" else 0,
        "room_type_reserved": room_type,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment,
        "repeated_guest": 1 if repeated_guest else 0,
        "no_of_previous_cancellations": prev_canc,
        "no_of_previous_bookings_not_canceled": prev_success,
        "avg_price_per_room": avg_price,
        "no_of_special_requests": special_requests
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
