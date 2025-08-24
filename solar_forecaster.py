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
        self.ann_model = None

    def get_lat_lon(self, city):
        """Convert city name to latitude and longitude using geopy."""
        try:
            location = self.geolocator.geocode(city)
            if location:
                return location.latitude, location.longitude
            raise ValueError(f"Location '{city}' not found.")
        except Exception as e:
            raise Exception(f"Geocoding error: {str(e)}")

    def fetch_forecast(self, lat, lon, tilt=30, azimuth=0, kwp=1):
        """Fetch solar energy forecast from Forecast.Solar API."""
        endpoint = f"{self.base_url}/{lat}/{lon}/{tilt}/{azimuth}/{kwp}"
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()['result']['watt_hours']
        except requests.RequestException as e:
            raise Exception(f"API error: {str(e)}")

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic solar data if no dataset is available."""
        np.random.seed(42)
        irradiance = np.random.uniform(200, 1000, n_samples)  # W/m²
        temperature = np.random.uniform(15, 35, n_samples)   # °C
        efficiency = 0.2  # 20% panel efficiency
        output_kwh = irradiance * 0.005 * efficiency + np.random.normal(0, 0.1, n_samples)
        df = pd.DataFrame({
            'Irradiance': irradiance,
            'Temperature': temperature,
            'Output_kWh': output_kwh
        })
        return df

    def train_ann(self, dataset_path=None):
        """Train ANN model on solar data (CSV or synthetic)."""
        if dataset_path:
            try:
                df = pd.read_csv(dataset_path)
                # Check and standardize column names
                required_cols = {'Irradiance', 'Temperature', 'Output_kWh'}
                available_cols = set(df.columns)
                if not required_cols.issubset(available_cols):
                    # Attempt to map common column names or calculate Output_kWh if missing
                    column_mapping = {
                        'GHI': 'Irradiance',
                        'Ambient Temperature': 'Temperature',
                        'Power Output': 'Output_kWh',
                        'Global Horizontal Irradiance': 'Irradiance'
                    }
                    df = df.rename(columns=lambda x: column_mapping.get(x, x))
                    if 'Output_kWh' not in df.columns:
                        # Calculate Output_kWh if irradiance is available (using 0.2 efficiency as default)
                        if 'Irradiance' in df.columns:
                            df['Output_kWh'] = df['Irradiance'] * 0.005 * 0.2  # Adjust factor based on data units
                        else:
                            raise ValueError("Dataset missing required columns or data to derive Output_kWh.")
                # Ensure all required columns are present after mapping
                if not all(col in df.columns for col in ['Irradiance', 'Temperature', 'Output_kWh']):
                    raise ValueError("Dataset must contain 'Irradiance', 'Temperature', 'Output_kWh' columns after mapping.")
            except FileNotFoundError:
                print("Dataset not found. Generating synthetic data.")
                df = self.generate_synthetic_data()
        else:
            # Default to NSRDB sample data (download and place locally or use API)
            default_path = "nsrdb_pretoria.csv"  # Placeholder; download from NSRDB for Pretoria
            try:
                df = pd.read_csv(default_path)
                # Map NSRDB columns (e.g., 'GHI' to 'Irradiance', 'Temperature' to 'Temperature', calculate 'Output_kWh')
                df = df.rename(columns={'GHI': 'Irradiance', 'Temperature': 'Temperature'})
                if 'Output_kWh' not in df.columns:
                    df['Output_kWh'] = df['Irradiance'] * 0.005 * 0.2  # Example calculation
            except FileNotFoundError:
                print("Default dataset not found. Generating synthetic data.")
                df = self.generate_synthetic_data()

        # Clean data (handle missing values)
        df = df.dropna(subset=['Irradiance', 'Temperature', 'Output_kWh'])

        X = df[['Irradiance', 'Temperature']].values
        y = df['Output_kWh'].values

        # Scale features
        X = self.scaler.fit_transform(X)

        # Build and train ANN
        self.ann_model = Sequential([
            Dense(10, activation='relu', input_shape=(2,)),
            Dense(1)
        ])
        self.ann_model.compile(optimizer='adam', loss='mse')
        self.ann_model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        print("ANN model trained successfully.")

    def predict_with_ann(self, irradiance, temperature):
        """Predict energy output using trained ANN."""
        if not self.ann_model:
            raise ValueError("ANN model not trained. Call train_ann first.")
        input_data = self.scaler.transform([[irradiance, temperature]])
        return self.ann_model.predict(input_data, verbose=0)[0][0]

    def plot_forecast(self, api_data, ann_predictions, timestamps):
        """Plot API and ANN forecasted energy outputs."""
        df_api = pd.DataFrame(list(api_data.items()), columns=['timestamp', 'watt_hours'])
        df_api['timestamp'] = pd.to_datetime(df_api['timestamp'])
        df_api['kwh'] = df_api['watt_hours'] / 1000  # Convert to kWh

        plt.figure(figsize=(12, 6))
        plt.plot(df_api['timestamp'], df_api['kwh'], marker='o', label='API Forecast')
        plt.plot(timestamps, ann_predictions, marker='x', linestyle='--', label='ANN Prediction')
        plt.title('Solar Energy Output Forecast Comparison')
        plt.xlabel('Time')
        plt.ylabel('Energy (kWh)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_forecast(self, city, dataset_path=None):
        """Main method to run forecast and compare API vs. ANN."""
        # Get location
        lat, lon = self.get_lat_lon(city)

        # Fetch API forecast
        api_forecast = self.fetch_forecast(lat, lon)

        # Train ANN
        self.train_ann(dataset_path)

        # Generate ANN predictions (use same timestamps as API for consistency)
        timestamps = list(api_forecast.keys())
        ann_predictions = []
        # For simplicity, use average irradiance/temperature or API-derived values
        for _ in timestamps:  # Replace with real weather data if available
            irradiance = np.random.uniform(200, 1000)  # Placeholder; ideally from API
            temperature = np.random.uniform(15, 35)
            ann_pred = self.predict_with_ann(irradiance, temperature)
            ann_predictions.append(ann_pred)

        # Plot results
        self.plot_forecast(api_forecast, ann_predictions, pd.to_datetime(timestamps))

# Example usage
if __name__ == "__main__":
    forecaster = SolarForecaster()
    city = input("Enter city (e.g., Pretoria): ")
    try:
        forecaster.run_forecast(city, dataset_path='nsrdb_pretoria.csv')  # Replace with your dataset path
    except Exception as e:
        print(f"Error: {str(e)}")