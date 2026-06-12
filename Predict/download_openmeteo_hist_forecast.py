import requests
import pandas as pd
# import os
from datetime import date, timedelta

# This function retrieves data from the OpenMeteo API
def get_open_meteo_data(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raises error for 400/404/500
        
        data = response.json()
        
        # Check if daily data exists in response
        if 'daily' not in data:
            print("Error: API returned valid JSON but no daily data.")
            return None

        daily = data['daily']
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'prcp_mm': daily['precipitation_sum'],
            'tmax_c': daily['temperature_2m_max'],
            'tmin_c': daily['temperature_2m_min']
        })
        
        # Add derived columns
        df['year'] = df['date'].dt.year
        df['yday'] = df['date'].dt.dayofyear
        
        # Clean (ERA5 sometimes has NaNs for the most recent days if slightly out of sync)
        df = df.dropna()
        
        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"URL attempted: {response.url}")
        return None
    except Exception as e:
        print(f"General Error: {e}")
        return None

# This function determines which OpenMeteo URL(s) to use (historical or forecast),
# calls get_open_meteo_data() to retrieve, and stitches the result(s) into a single dataframe
def fetch_open_meteo_history(lat, lon, start_date, end_date):
    """
    Fetches historical weather data from Open-Meteo (ERA5 Reanalysis).
    Automatically caps the 'end_date' to yesterday to avoid API 400 errors.
    """
    # Archive URL for historical data
    url_archive = "https://archive-api.open-meteo.com/v1/archive"
    # Forecast URL for future data
    url_forecast = "https://api.open-meteo.com/v1/forecast"

    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Determine if we need to get archive or forecast data
    get_archive = False
    get_forecast = False
    if start_date < date.today():
        get_archive = True
        start_date_archive = start_date
        if end_date < date.today():
            end_date_archive = end_date
        else:
            end_date_archive = date.today() - timedelta(days=1)
            get_forecast = True
            start_date_forecast = date.today()
            end_date_forecast = end_date
    else:
        get_forecast = True
        start_date_forecast = start_date
        end_date_forecast = end_date

    if get_archive:
        url = url_archive
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_archive,
            "end_date": end_date_archive,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "America/Los_Angeles"
        }

        print(f"Fetching Open-Meteo history ({start_date_archive} to {end_date_archive})...")
        df_archive = get_open_meteo_data(url, params)

    
    if get_forecast:
        url = url_forecast
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_forecast,
            "end_date": end_date_forecast,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "America/Los_Angeles"
        }
        print(f"Fetching Open-Meteo forecast ({start_date_forecast} to {end_date_forecast})...")
        df_forecast = get_open_meteo_data(url, params)

    # Concatenate df_archive and df_forecast, marking source
    frames = []
    if 'df_archive' in locals() and df_archive is not None:
        df_archive = df_archive.copy()
        df_archive['is_forecast'] = False
        frames.append(df_archive)
    if 'df_forecast' in locals() and df_forecast is not None:
        df_forecast = df_forecast.copy()
        df_forecast['is_forecast'] = True
        frames.append(df_forecast)
    if frames:
        df_concat = pd.concat(frames, ignore_index=True)
    else:
        df_concat = None

    
    return df_concat



# --- TESTING ---

# # Pt Reyes Bear Valley Visitor Center Lat/Lon
# LAT = 38.0396
# LON = -122.7984
# # Dates
# startdate = '2026-06-12'
# enddate = '2026-06-27'

# # We can safely leave 2026 here now; the script will auto-correct it to today's date
# history_df = fetch_open_meteo_history(LAT, LON, startdate, enddate)

# print(history_df)