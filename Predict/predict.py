# This script updates datasets and makes predictions using the trained model

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
# from datetime import date, timedelta
from download_openmeteo_hist_forecast import fetch_open_meteo_hist_forecast
import pickle
import os

# Set directory to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CONSTANTS
# ================================================
# Model file name
current_model_file = 'model_2026-06-23.pkl'
# Porcini index max bound
porcini_index_max = 7.5
# Prediction features
features_list = [
    # observer effort features
    'tmax_c', 'tmin_c', 'prcp_mm', 'is_weekend', #'dayofweek',
    # precip features
    '14_day_prcp_mm', '30_day_prcp_mm', '60_day_prcp_mm', 
    # 'days_since_last_precip_over_1mm', 'days_since_last_precip_over_3mm', 
    'days_since_last_precip_over_5mm',
    # temperature features
    '7day_tmax_c', '7day_tmin_c', '14day_tmax_c', '14day_tmin_c', '14-7_tmax_c', '14-7_tmin_c',
    # time features (exclude - mushrooms don't know the date)
    'sin_time', 'cos_time'
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


# BUILD WEATHER FEATURES
# ================================================
def build_weather_features(startdate, enddate):
    # Bear Valley Visitor Center Lat/Lon
    LAT = 38.0396
    LON = -122.7984
    # Dates
    # startdate = dt.date.today() - dt.timedelta(days=365) # Grab a year of historical weather data
    # enddate = dt.date.today() + dt.timedelta(days=14) # Two weeks forecast
    
    # Fetch historical and forecast weather data (including 365 days of history prior to prediction start date)
    weather_df = fetch_open_meteo_hist_forecast(LAT, LON, startdate - dt.timedelta(days=365), enddate)

    # Build features
    print("Building weather features...")
    weather_df['is_weekend'] = weather_df['date'].dt.dayofweek.isin([5,6])
    weather_df['14_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=14, closed='left').sum()
    weather_df['30_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=30, closed='left').sum()
    weather_df['60_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=60, closed='left').sum()
    weather_df['7day_tmax_c'] = weather_df['tmax_c'].rolling(window=7, closed='left').mean()
    weather_df['7day_tmin_c'] = weather_df['tmin_c'].rolling(window=7, closed='left').mean()
    weather_df['14day_tmax_c'] = weather_df['tmax_c'].rolling(window=14, closed='left').mean()
    weather_df['14day_tmin_c'] = weather_df['tmin_c'].rolling(window=14, closed='left').mean()
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

    # weather_df['days_since_july_1'] =  ((weather_df['yday'] - 182) % 365).astype(int)
    weather_df['sin_time'] = np.sin(2 * np.pi * weather_df['yday'] / 365)
    weather_df['cos_time'] = np.cos(2 * np.pi * weather_df['yday'] / 365)

    # Restrict to startdate to enddate (remove weather history used only for building features)
    weather_df = weather_df[(weather_df['date'] >= startdate) & (weather_df['date'] <= enddate)]

    return weather_df

    
# MAKE PREDICTIONS
# ================================================
def make_predictions(start_date=None, end_date=None):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Grab weather data, including 365 days of history prior to prediction start date
    print("Calling weather data pipeline...")
    weather_df = build_weather_features(start_date, end_date)

    # IMPORTANT: IMPOSE "PERFECT DAY" FEATURES
    print("Imposing perfect day features...")
    prediction_df = weather_df.copy()
    for feature, value in perfect_day_features.items():
        prediction_df[feature] = value

    # Load trained model
    with open(current_model_file, 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(prediction_df[features_list])
    prediction_df['predicted_count'] = predictions

    # Normalize predicted count to 0-100 **** Fill this in once settled on a model - Need a max bound on the predicted count
    prediction_df['porcini_index'] = prediction_df['predicted_count']/porcini_index_max * 100

    # Bring back true values of perfect day features (useful for displaying, e.g. rain along with porcini forecast)
    for feature in perfect_day_features.keys():
        prediction_df[feature + '_true'] = weather_df[feature]

    return prediction_df


# SAVE PREDICTIONS
# ================================================
prediction_df = make_predictions(dt.date.today(), dt.date.today() + dt.timedelta(days=14))
# prediction_df = make_predictions(dt.date(2023, 12, 24), dt.date(2023, 12, 24) + dt.timedelta(days=14))
prediction_df.to_csv('../webapp_streamlit/predictions.csv', index=False)