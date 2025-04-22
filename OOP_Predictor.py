import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class HotelBookingPredictor:
    def __init__(self, model_path, scaler_path=None, encoder_path=None):
        # Load the trained model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load the saved scaler
        if scaler_path is not None:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = StandardScaler()  # Default scaler (not fitted)

        # Load the saved encoders (dictionary of LabelEncoders)
        if encoder_path is not None:
            with open(encoder_path, "rb") as f:
                self.encoder = pickle.load(f)  # Dictionary of LabelEncoders
        else:
            self.encoder = {}  # Default empty dictionary

        # Define categorical and numerical features
        self.cat_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        self.num_features = [
            'lead_time', 'avg_price_per_room',
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
            'no_of_previous_bookings_not_canceled', 'no_of_previous_cancellations', 'no_of_special_requests'
        ]

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        # Convert input data to a DataFrame
        df = pd.DataFrame([input_data])

        # Encode categorical features using the loaded encoders
        for col in self.cat_features:
            if col in self.encoder:
                # Use transform (not fit_transform) since the encoder is already fitted
                df[col] = self.encoder[col].transform([df[col].iloc[0]])[0]
            else:
                raise ValueError(f"Encoder for column '{col}' not found in the loaded encoders.")

        # Scale numerical features using the loaded scaler
        df[self.num_features] = self.scaler.transform(df[self.num_features])

        return df

    def predict(self, input_data: dict) -> int:
        # Preprocess the input data
        processed = self.preprocess_input(input_data)

        # Make a prediction using the loaded model
        prediction = self.model.predict(processed)[0]

        # Return the prediction as an integer
        return int(prediction)