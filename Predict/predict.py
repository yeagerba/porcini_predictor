# This script updates datasets and makes predictions using the trained model

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
# from datetime import date, timedelta
from download_openmeteo_hist_forecast import fetch_open_meteo_hist_forecast
import pickle

# CONSTANTS
# ================================================
# Model file name
current_model_file = 'model_2026-06-08.pkl'
# Prediction features
features_list = [
    'tmax_c', 'tmin_c', 'prcp_mm', 'is_weekend',
    '14_day_prcp_mm', '30_day_prcp_mm', '60_day_prcp_mm', 
    'days_since_last_precip_over_1mm', 'days_since_last_precip_over_3mm', 'days_since_last_precip_over_5mm',
    '7day_tmax_c', '7day_tmin_c', '14day_tmax_c', '14day_tmin_c', '14-7_tmax_c', '14-7_tmin_c',
    'yday'
]
# Perfect day features
# (We are predicting the likelihood of mushroom fruiting, NOT the likelihood of mushroom observations, 
# so we remove the affects of the day of the week, temperature, and precipitation.)
perfect_day_features = {
    'is_weekend': True,
    'tmax_c': 12,
    'tmin_c': 8,
    'prcp_mm': 0
}


# FETCH WEATHER DATA
# ================================================
def fetch_weather_data():
    # Bear Valley Visitor Center Lat/Lon
    LAT = 38.0396
    LON = -122.7984
    # Dates
    startdate = dt.date.today() - dt.timedelta(days=365) # Grab a year of historical weather data
    enddate = dt.date.today() + dt.timedelta(days=14) # Two weeks forecast
    # Fetch historical and forecast weather data
    weather_df = fetch_open_meteo_hist_forecast(LAT, LON, startdate, enddate)

    # print(weather_df)
    # # print(weather_df.prcp_mm.max())
    # print(weather_df.columns)

    # Build features
    weather_df['is_weekend'] = weather_df['date'].dt.dayofweek.isin([5,6])
    weather_df['14_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=14).mean()
    weather_df['30_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=30).mean()
    weather_df['60_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=60).mean()
    weather_df['7day_tmax_c'] = weather_df['tmax_c'].rolling(window=7).mean()
    weather_df['7day_tmin_c'] = weather_df['tmin_c'].rolling(window=7).mean()
    weather_df['14day_tmax_c'] = weather_df['tmax_c'].rolling(window=14).mean()
    weather_df['14day_tmin_c'] = weather_df['tmin_c'].rolling(window=14).mean()
    weather_df['14-7_tmax_c'] = weather_df['14day_tmax_c'] - weather_df['7day_tmax_c']
    weather_df['14-7_tmin_c'] = weather_df['14day_tmin_c'] - weather_df['7day_tmin_c']

    # Function to build days since last precip >= threshold
    def build_days_since_last_precip(weather_df, precip_threshold_mm):
        last_precip_date = None
        days_since_last_precip = []
        for idx, row in weather_df.iterrows():
            if last_precip_date is None:
                days_since_last_precip.append(None)
            else:
                days_since_last_precip.append((row['date'] - last_precip_date).days)
        

            if row['prcp_mm'] >= precip_threshold_mm:
                last_precip_date = row['date']
        
        return days_since_last_precip

    weather_df['days_since_last_precip_over_1mm'] = build_days_since_last_precip(weather_df, 1)
    weather_df['days_since_last_precip_over_3mm'] = build_days_since_last_precip(weather_df, 3)
    weather_df['days_since_last_precip_over_5mm'] = build_days_since_last_precip(weather_df, 5)

    return weather_df

    
# MAKE PREDICTIONS
# ================================================
def make_predictions():
    weather_df = fetch_weather_data()

    # IMPORTANT: IMPOSE "PERFECT DAY" FEATURES
    prediction_df = weather_df.copy()
    for feature, value in perfect_day_features.items():
        prediction_df[feature] = value

    # Load trained model
    with open(current_model_file, 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    predictions = model.predict(prediction_df[features_list])


    prediction_df['predicted_count'] = predictions

    # Bring back true values of perfect day features (useful for displaying, e.g. rain along with porcini forecast)
    for feature in perfect_day_features.keys():
        prediction_df[feature + '_true'] = weather_df[feature]

    return prediction_df


# SAVE PREDICTIONS
# ================================================
# prediction_df = make_predictions()
# prediction_df.to_csv('predictions.csv', index=False)