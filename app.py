import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class SolarForecaster:
    def __init__(self):
        self.base_url = "https://api.forecast.solar/estimate"
        self.geolocator = Nominatim(user_agent="solar_forecaster")
        self.scaler = StandardScaler()

    def get_lat_lon(self, city):
        try:
            location = self.geolocator.geocode(city)
            if location:
                return location.latitude, location.longitude
            raise ValueError(f"Location '{city}' not found.")
        except Exception as e:
            raise Exception(f"Geocoding error: {str(e)}")

    def fetch_forecast(self, lat, lon, tilt=30, azimuth=0, kwp=1):
        endpoint = f"{self.base_url}/{lat}/{lon}/{tilt}/{azimuth}/{kwp}"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()['result']['watt_hours']
        except requests.RequestException as e:
            raise Exception(f"API error: {str(e)}")

    def generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        irradiance = np.random.uniform(200, 1000, n_samples)
        temperature = np.random.uniform(15, 35, n_samples)
        efficiency = 0.2
        output_kwh = irradiance * 0.005 * efficiency + np.random.normal(0, 0.1, n_samples)
        df = pd.DataFrame({
            'Irradiance': irradiance,
            'Temperature': temperature,
            'Output_kWh': output_kwh
        })
        return df

@st.cache_resource
def train_ann(_forecaster, dataset_path=None, uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif dataset_path:
        df = pd.read_csv(dataset_path)
    else:
        df = forecaster.generate_synthetic_data()

    if not all(col in df.columns for col in ['Irradiance', 'Temperature', 'Output_kWh']):
        st.error("Dataset must contain 'Irradiance', 'Temperature', 'Output_kWh' columns.")
        return None

    X = df[['Irradiance', 'Temperature']].values
    y = df['Output_kWh'].values

    X = forecaster.scaler.fit_transform(X)

    model = Sequential([
        Dense(10, activation='relu', input_shape=(2,)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    st.success("ANN model trained successfully.")
    return model

def predict_with_ann(model, forecaster, irradiance, temperature):
    if not model:
        st.error("ANN model not trained.")
        return None
    input_data = forecaster.scaler.transform([[irradiance, temperature]])
    return model.predict(input_data, verbose=0)[0][0]

def plot_forecast(api_data, ann_predictions, timestamps):
    df_api = pd.DataFrame(list(api_data.items()), columns=['timestamp', 'watt_hours'])
    df_api['timestamp'] = pd.to_datetime(df_api['timestamp'])
    df_api['kwh'] = df_api['watt_hours'] / 1000

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_api['timestamp'], df_api['kwh'], marker='o', label='API Forecast')
    ax.plot(timestamps, ann_predictions, marker='x', linestyle='--', label='ANN Prediction')
    ax.set_title('Solar Energy Output Forecast Comparison')
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy (kWh)')
    ax.grid(True)
    ax.tick_params(rotation=45)
    ax.legend()
    return fig

# Streamlit UI
st.title("Solar Energy Output Forecaster")
st.markdown("Enter a location to forecast solar energy output. Uses Forecast.Solar API and a simple ANN for predictions.")

city = st.text_input("City (e.g., Pretoria)", "Pretoria")
tilt = st.slider("Panel Tilt (degrees)", 0, 90, 30)
azimuth = st.slider("Panel Azimuth (degrees, 0=south)", -180, 180, 0)
kwp = st.number_input("System Capacity (kWp)", min_value=0.1, value=1.0)

# Dataset options
use_synthetic = st.checkbox("Use Synthetic Data for ANN Training", value=True)
uploaded_file = st.file_uploader("Upload CSV Dataset (optional)", type="csv") if not use_synthetic else None

# For ANN predictions: Allow average weather input
avg_irradiance = st.number_input("Average Irradiance (W/m²) for ANN", min_value=0.0, value=600.0)
avg_temperature = st.number_input("Average Temperature (°C) for ANN", min_value=-10.0, value=25.0)

if st.button("Run Forecast"):
    with st.spinner("Fetching data and running predictions..."):
        forecaster = SolarForecaster()
        try:
            lat, lon = forecaster.get_lat_lon(city)
            api_forecast = forecaster.fetch_forecast(lat, lon, tilt, azimuth, kwp)

            # Train ANN
            ann_model = train_ann(forecaster, dataset_path="nsrdb_pretoria.csv" if not uploaded_file else uploaded_file)

            # Generate ANN predictions (one per API timestamp, using average weather for simplicity)
            timestamps = list(api_forecast.keys())
            ann_predictions = [predict_with_ann(ann_model, forecaster, avg_irradiance, avg_temperature) for _ in timestamps]

            # Plot
            fig = plot_forecast(api_forecast, ann_predictions, pd.to_datetime(timestamps))
            st.pyplot(fig)

            st.success(f"Forecast for {city} (Lat: {lat:.2f}, Lon: {lon:.2f}) completed.")
        except Exception as e:
            st.error(f"Error: {str(e)}")