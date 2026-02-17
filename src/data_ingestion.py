import sys
import os

# --- PATH FIX: Add Project Root to System Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# --------------------------------------------------

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
from src.database import AQIDatabase # Now this works

def fetch_historical_data():
    """
    Fetches historical AQI data for Karachi from Open Meteo API.
    """
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    latitude = 24.8607
    longitude = 67.0011
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=6*30)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    }

    print(f"Fetching data from {start_date} to {end_date}...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    # Extract variables logic...
    hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
    hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
    hourly_co = hourly.Variables(2).ValuesAsNumpy()
    hourly_no2 = hourly.Variables(3).ValuesAsNumpy()
    hourly_so2 = hourly.Variables(4).ValuesAsNumpy()
    hourly_o3 = hourly.Variables(5).ValuesAsNumpy()
    hourly_us_aqi = hourly.Variables(6).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["pm10"] = hourly_pm10
    hourly_data["pm2_5"] = hourly_pm2_5
    hourly_data["carbon_monoxide"] = hourly_co
    hourly_data["nitrogen_dioxide"] = hourly_no2
    hourly_data["sulphur_dioxide"] = hourly_so2
    hourly_data["ozone"] = hourly_o3
    hourly_data["us_aqi"] = hourly_us_aqi

    return pd.DataFrame(data = hourly_data)

if __name__ == "__main__":
    try:
        print("Starting data ingestion process...")
        df = fetch_historical_data()
        print(f"✅ Fetched {len(df)} records from Open-Meteo API.")
        
        if not df.empty:
            print("Connecting to MongoDB Atlas...")
            db = AQIDatabase()
            db.insert_data(df)
            print("✅ Data successfully ingested into MongoDB.")
        else:
            print("⚠️ No data fetched.")

    except Exception as e:
        print(f"❌ Data ingestion failed: {e}")