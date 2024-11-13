import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit


# Make `src` dir can be imported
project_root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root_path)

from src.utils import choose_device
from src.train import generate_scores_by_xgb
from src.preprocess import preprocess
from src.feature_engineering import (
    calculate_pressure_diff,
    create_sinusoidal_transformation_by_number,
    create_sinusoidal_transformation_year_month_day,
    create_time_features,
)

day_gap = 24 * 60
raw_data_path = "data/processed_data/combined_data.csv"
device_name = "L8"

def read_raw_data_and_sort(raw_data_path: str, sort_by_cols: list = ["device", "datetime"]) -> pd.DataFrame:
    df_raw_data = pd.read_csv(raw_data_path, parse_dates=["datetime"])
    # make sure the sorting is correct
    df_raw_data = df_raw_data.sort_values(by=sort_by_cols).reset_index(drop=True)
    return df_raw_data

def generate_sinusoidal_date_features(start_date, end_date, col_name):
    dates = pd.date_range(start=start_date, end=end_date)

    # Create DataFrame and extract year, month, and day into separate columns
    df_dates = pd.DataFrame({
        "year": dates.year, 
        "month": dates.month, 
        "day": dates.day
    })

    return create_sinusoidal_transformation_year_month_day(df_dates, col_name, "year", "month", "day", 12)

def split_train_test(df, plot=False):
    _, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

    fold = 0
    for train_idx, val_idx in tss.split(df):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[val_idx]

        if plot:
            df_train["power"].plot(
                ax=axs[fold],
                label="Training Set",
                title=f"Data Train/Test Split Fold {fold}",
            )
            df_test["power"].plot(ax=axs[fold], label="Test Set")
            axs[fold].axvline(df_test.index.min(), color="black", ls="--")

            fold += 1

    return df_train, df_test

def get_test_size(days):
    # since the data is one minute one row
    return days * 24 * 60


df_raw_data = read_raw_data_and_sort(raw_data_path, sort_by_cols=["device", "datetime"])

# parameterize the device for testing conveniently
df_device = choose_device(df_raw_data, device_name)

columns_to_standardize = ["windspeed", "temperature", "humidity", "sunlight"]
df_standardized = preprocess(df_device, columns_to_standardize)

## Feature engineering
df_fe_result = calculate_pressure_diff(df_standardized, column="pressure")

# create sinusodial month mapping
month_numbers = list(range(1, 13))
df_months = pd.DataFrame(month_numbers, columns=["month"])
df_months = create_sinusoidal_transformation_by_number(df_months, "month", 12)

# Generate date range
general_dates_params = {
    "start_date": "2023-01-01", 
    "end_date": "2025-12-31",
    "col_name": "general_ymd"
}
lunar_dates_params = {
    "start_date": "2023-02-04", 
    "end_date": "2025-02-03",
    "col_name": "lunar_ymd"
}
df_general_dates = generate_sinusoidal_date_features(**general_dates_params)
df_lunar_dates = generate_sinusoidal_date_features(**lunar_dates_params)

# Aligns with a specific lunar calendar period
df_fe_result = create_time_features(df_fe_result, "datetime")
df_fe_result_sinusoidal_time = pd.merge(
    df_fe_result, df_general_dates, how="left", on=["year", "month", "day"]
)

# input data
drop_cols = ["locationcode", "year", "pressure"]
df_fe_result_sinusoidal_time = df_fe_result_sinusoidal_time.drop(drop_cols, axis=1)
df_fe_result_sinusoidal_time = df_fe_result_sinusoidal_time.set_index("datetime")

## Train/Test split
day_gap = 24 * 60
drop_cols = ["power", "device", "date"]
TARGET = "power"
n_splits = 5

tss = TimeSeriesSplit(n_splits=n_splits, test_size=get_test_size(5), gap=day_gap)
df_train, df_test = split_train_test(df=df_fe_result_sinusoidal_time, plot=True)

## Train Using Cross Validation
tss = TimeSeriesSplit(n_splits=n_splits, test_size=get_test_size(2), gap=day_gap)

df = df_fe_result_sinusoidal_time
scores = generate_scores_by_xgb(df, tss)
print(f"Total absolute error {np.mean(scores):.2f}")
