import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta
from src.database import AQIDatabase

def fetch_historical_data():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    latitude = 24.8607
    longitude = 67.0011
    
    # FIX: Changed from 7 days to 4 days to satisfy API limits
    # The API limit is usually 5-6 days, so 4 is safe.
    end_date = datetime.now().date() + timedelta(days=4)
    start_date = datetime.now().date() - timedelta(days=6*30)

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
    
    # Extract data
    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    
    for i, col in enumerate(["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]):
        hourly_data[col] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data = hourly_data)

if __name__ == "__main__":
    try:
        print("Starting data ingestion...")
        df = fetch_historical_data()
        print(f"✅ Fetched {len(df)} records (History + Forecast).")
        
        if not df.empty:
            db = AQIDatabase()
            db.insert_data(df)
            print("✅ Data ingested into MongoDB.")
    except Exception as e:
        print(f"❌ Error: {e}")
        