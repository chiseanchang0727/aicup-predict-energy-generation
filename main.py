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

from src.utils import read_config, choose_device
from src.train import generate_scores_by_xgb
from src.preprocess import preprocess

from src.feature_engineering import feature_engineering


######################################################################

config_file = 'test_3_L10.json'
configs = read_config(os.path.join('./test_configs/', config_file))

device_name = configs['device_name']
cols_for_drop = configs['cols_for_drop']


######################################################################


day_gap = 24 * 60
raw_data_path = "data/processed_data/combined_data.csv"


def read_raw_data_and_sort(raw_data_path: str, sort_by_cols: list = ["device", "datetime"]) -> pd.DataFrame:
    df_raw_data = pd.read_csv(raw_data_path, parse_dates=["datetime"])
    # make sure the sorting is correct
    df_raw_data = df_raw_data.sort_values(by=sort_by_cols).reset_index(drop=True)
    return df_raw_data


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


df_preprocessing = preprocess(df_device, cols_for_drop)

# feature engineering
df_fe_result = feature_engineering(df=df_preprocessing)



## Train/Test split
day_gap = 24 * 60
drop_cols = ["power", "device", "date"]
TARGET = "power"
n_splits = 5

## Train Using Cross Validation
tss = TimeSeriesSplit(n_splits=n_splits, test_size=get_test_size(5), gap=day_gap)
df_train, df_test = split_train_test(df=df_fe_result, plot=True)

df = df_fe_result
scores = generate_scores_by_xgb(df, tss)
print(f"Total absolute error: {np.mean(scores):.2f}")
