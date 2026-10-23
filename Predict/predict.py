# This script updates datasets and makes predictions using the trained model

import pandas as pd
import numpy as np
import geopandas as gpd
import datetime as dt
# from datetime import date, timedelta
from download_openmeteo_hist_forecast import fetch_open_meteo_hist_forecast
import pickle
import os
import xgboost as xgb

# Set directory to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CONSTANTS
# ================================================
# Model file name
current_classifier_file = 'xgb_classifier_king_bolete_2026-10-23.json'
# current_regressor_file = 'xgb_regressor_2026-10-22.json'
# Porcini index max bound
porcini_index_max = 0.575640*1.1 # 10% buffer
# Prediction features
features_list = [
    # precip features
    'prcp_mm_sum_4d', 'prcp_mm_sum_7d', 'prcp_mm_sum_14d', 'prcp_mm_sum_28d', 'prcp_mm_sum_56d', 'prcp_mm_sum_84d', 
    # temperature features
    'tmax_c_mean_5d', 'tmax_c_mean_12d', 'tmax_c_mean_26d', 'tmax_c_mean_60d', 
    'tmin_c_mean_5d', 'tmin_c_mean_12d', 'tmin_c_mean_26d', 'tmin_c_mean_60d', 
    # time features
    'sin_time', 'cos_time',
    # observer effort features
    'tmax_c', 'tmin_c', 'prcp_mm', 'cos_dayofweek', 'sin_dayofweek', 'year'
]
# Perfect day features
# (We are predicting the likelihood of mushroom fruiting, NOT the likelihood of mushroom observations, 
# so we remove the affects of the day of the week, temperature, and precipitation.)
perfect_day_features = {
    'cos_dayofweek' : np.cos(2 * np.pi * 6 / 7), # Saturday
    'sin_dayofweek' : np.sin(2 * np.pi * 6 / 7), # Saturday
    'tmax_c' : 12,
    'tmin_c' : 8,
    'prcp_mm' : 0,
    'year' : 2023
}


# BUILD WEATHER FEATURES
# ================================================

# DEPRECATED - PRESERVED FOR REFERENCE - USE XGBOOST HURDLE MODEL INSTEAD
# def build_weather_features(startdate, enddate):
#     # Bear Valley Visitor Center Lat/Lon
#     LAT = 38.0396
#     LON = -122.7984
#     # Dates
#     # startdate = dt.date.today() - dt.timedelta(days=365) # Grab a year of historical weather data
#     # enddate = dt.date.today() + dt.timedelta(days=14) # Two weeks forecast
    
#     # Fetch historical and forecast weather data (including 365 days of history prior to prediction start date)
#     weather_df = fetch_open_meteo_hist_forecast(LAT, LON, startdate - dt.timedelta(days=365), enddate)

#     # Build features
#     print("Building weather features...")
#     weather_df['is_weekend'] = weather_df['date'].dt.dayofweek.isin([5,6])
#     weather_df['14_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=14, closed='left').sum()
#     weather_df['30_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=30, closed='left').sum()
#     weather_df['60_day_prcp_mm'] = weather_df['prcp_mm'].rolling(window=60, closed='left').sum()
#     weather_df['7day_tmax_c'] = weather_df['tmax_c'].rolling(window=7, closed='left').mean()
#     weather_df['7day_tmin_c'] = weather_df['tmin_c'].rolling(window=7, closed='left').mean()
#     weather_df['14day_tmax_c'] = weather_df['tmax_c'].rolling(window=14, closed='left').mean()
#     weather_df['14day_tmin_c'] = weather_df['tmin_c'].rolling(window=14, closed='left').mean()
#     weather_df['14-7_tmax_c'] = weather_df['14day_tmax_c'] - weather_df['7day_tmax_c']
#     weather_df['14-7_tmin_c'] = weather_df['14day_tmin_c'] - weather_df['7day_tmin_c']

#     # Function to build days since last precip >= threshold
#     def build_days_since_last_precip(weather_df, precip_threshold_mm):
#         last_precip_date = None
#         days_since_last_precip = []
#         for idx, row in weather_df.iterrows():
#             if last_precip_date is None:
#                 days_since_last_precip.append(None)
#             else:
#                 days_since_last_precip.append((row['date'] - last_precip_date).days)
        

#             if row['prcp_mm'] >= precip_threshold_mm:
#                 last_precip_date = row['date']
        
#         return days_since_last_precip

#     weather_df['days_since_last_precip_over_1mm'] = build_days_since_last_precip(weather_df, 1)
#     weather_df['days_since_last_precip_over_3mm'] = build_days_since_last_precip(weather_df, 3)
#     weather_df['days_since_last_precip_over_5mm'] = build_days_since_last_precip(weather_df, 5)

#     # weather_df['days_since_july_1'] =  ((weather_df['yday'] - 182) % 365).astype(int)
#     weather_df['sin_time'] = np.sin(2 * np.pi * weather_df['yday'] / 365)
#     weather_df['cos_time'] = np.cos(2 * np.pi * weather_df['yday'] / 365)

#     # Restrict to startdate to enddate (remove weather history used only for building features)
#     weather_df = weather_df[(weather_df['date'] >= startdate) & (weather_df['date'] <= enddate)]

#     return weather_df

def build_weather_features(startdate, enddate):
    # Bear Valley Visitor Center Lat/Lon
    LAT = 38.0396
    LON = -122.7984
    # Dates
    # startdate = dt.date.today() - dt.timedelta(days=365) # Grab a year of historical weather data
    # enddate = dt.date.today() + dt.timedelta(days=14) # Two weeks forecast
    
    # Fetch historical and forecast weather data (including 365 days of history prior to prediction start date)
    weather_df = fetch_open_meteo_hist_forecast(LAT, LON, startdate - dt.timedelta(days=365), enddate)

    if len(weather_df) < (enddate - (startdate - dt.timedelta(days=365))).days + 1:
        raise ValueError(f"Weather data fetch failed. Only {len(weather_df)} days of weather data available, but {(enddate - (startdate - dt.timedelta(days=365))).days + 1} days are needed.")

    # Build features
    weather_df['prcp_mm_sum_4d'] = weather_df['prcp_mm'].rolling(window=4, closed='left').sum()
    weather_df['prcp_mm_sum_7d'] = weather_df['prcp_mm'].rolling(window=7, closed='left').sum()
    weather_df['prcp_mm_sum_14d'] = weather_df['prcp_mm'].rolling(window=14, closed='left').sum()
    weather_df['prcp_mm_sum_28d'] = weather_df['prcp_mm'].rolling(window=28, closed='left').sum()
    weather_df['prcp_mm_sum_56d'] = weather_df['prcp_mm'].rolling(window=56, closed='left').sum()
    weather_df['prcp_mm_sum_84d'] = weather_df['prcp_mm'].rolling(window=84, closed='left').sum()
    weather_df['tmax_c_mean_5d'] = weather_df['tmax_c'].rolling(window=5, closed='left').mean()
    weather_df['tmax_c_mean_12d'] = weather_df['tmax_c'].rolling(window=12, closed='left').mean()
    weather_df['tmax_c_mean_26d'] = weather_df['tmax_c'].rolling(window=26, closed='left').mean()
    weather_df['tmax_c_mean_60d'] = weather_df['tmax_c'].rolling(window=60, closed='left').mean()
    weather_df['tmin_c_mean_5d'] = weather_df['tmin_c'].rolling(window=5, closed='left').mean()
    weather_df['tmin_c_mean_12d'] = weather_df['tmin_c'].rolling(window=12, closed='left').mean()
    weather_df['tmin_c_mean_26d'] = weather_df['tmin_c'].rolling(window=26, closed='left').mean()
    weather_df['tmin_c_mean_60d'] = weather_df['tmin_c'].rolling(window=60, closed='left').mean()
    weather_df['sin_time'] = np.sin(2 * np.pi * weather_df['yday'] / 365)
    weather_df['cos_time'] = np.cos(2 * np.pi * weather_df['yday'] / 365)
    # weather_df['tmax_c'] = 
    # weather_df['tmin_c'] = 
    # weather_df['prcp_mm'] = 
    weather_df['cos_dayofweek'] = weather_df['date'].dt.dayofweek.apply(lambda x: np.cos(2 * np.pi * x / 7))
    weather_df['sin_dayofweek'] = weather_df['date'].dt.dayofweek.apply(lambda x: np.sin(2 * np.pi * x / 7))
    # weather_df['year'] = 

    return weather_df

# MAKE PREDICTIONS
# ================================================
def make_predictions(start_date=None, end_date=None, smoothing_window=0):
    start_date = pd.to_datetime(start_date)
    max_end_date = dt.date.today() + dt.timedelta(days=15) - dt.timedelta(days=int(np.floor(smoothing_window / 2)))
    if end_date > max_end_date:
        print(f"***Warning: End date ({end_date}) is greater than max end date ({max_end_date}) of OpenMeteo data. Setting end date to {max_end_date}.")
        num_days = (max_end_date - start_date.date()).days + 1
        print(f"***This will limit the length of the forecast to {num_days} days.")
        end_date = max_end_date
    end_date = pd.to_datetime(end_date)

    # Add buffer days to start_date and end_date to ensure we have enough weather data for the smoothing window
    start_date_model = start_date - dt.timedelta(days=int(np.floor(smoothing_window / 2)))
    end_date_model = end_date + dt.timedelta(days=int(np.floor(smoothing_window / 2)))

    # Grab weather data, including 365 days of history prior to prediction start date
    print("Calling weather data pipeline...")
    weather_df = build_weather_features(start_date_model, end_date_model)

    # IMPORTANT: IMPOSE "PERFECT DAY" FEATURES
    print("Imposing perfect day features...")
    prediction_df = weather_df[(weather_df['date'] >= start_date_model) & (weather_df['date'] <= end_date_model)].copy()
    for feature, value in perfect_day_features.items():
        prediction_df[feature] = value

    # Load trained models (XGBoost expects a file path, not an open file handle)
    classifier = xgb.XGBClassifier()
    classifier.load_model(current_classifier_file)
    # regressor = xgb.XGBRegressor()
    # regressor.load_model(current_regressor_file)

    # Make predictions
    print("Making predictions...")
    classifier_predictions = classifier.predict_proba(prediction_df[features_list])
    prediction_df['predicted_occurence'] = classifier_predictions[:, 1]
    # regressor_predictions = regressor.predict(prediction_df[features_list])
    # prediction_df['predicted_count'] = regressor_predictions

    prediction_df['predicted_final'] = prediction_df['predicted_occurence'] #* prediction_df['predicted_count']

    # Normalize predicted count to 0-100 **** Fill this in once settled on a model - Need a max bound on the predicted count
    prediction_df['porcini_index'] = prediction_df['predicted_final'].rolling(window=smoothing_window, center=True).mean()/porcini_index_max * 100

    # Bring back true values of perfect day features (useful for displaying, e.g. rain along with porcini forecast)
    for feature in perfect_day_features.keys():
        prediction_df[feature + '_true'] = weather_df[feature]

    print(prediction_df.head())

    return prediction_df[(pd.to_datetime(prediction_df['date']) >= start_date) & (pd.to_datetime(prediction_df['date']) <= end_date)]


# SAVE PREDICTIONS
# ================================================
prediction_df = make_predictions(dt.date.today(), dt.date.today() + dt.timedelta(days=14), smoothing_window=5)
print(prediction_df.head())
print(prediction_df.columns)
# prediction_df = make_predictions(dt.date(2023, 12, 24), dt.date(2023, 12, 24) + dt.timedelta(days=14))
prediction_df.to_csv('predictions.csv', index=False)