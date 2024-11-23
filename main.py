import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit

from src.utils import read_config, choose_device
from src.train import train_and_valid
from src.preprocess import preprocess
from src.feature_engineering import feature_engineering
from src.standardization import standardization

######################################################################

config_file = './Sean/test_11_L10_rolling.json'
configs = read_config(os.path.join('./test_configs/', config_file))

test_name = config_file.split('/')[2].split('.')[0]
print(f"{test_name} starts.")
print("configs: ", configs)
# test_name = 'test_6_L10_sunlight_sim'

device_name = configs['device_name']
cols_for_drop = configs['cols_for_drop']

# preprocess_config = configs['preprocess_config]
fe_config = configs['fe_config']

pred_result_ouput = configs['pred_result_ouput']

n_splits = configs['n_splits']

######### test these later

day_gap = 24 * 60
invalid_cols_for_training = ["device", "datetime", "date"]

######################################################################

target = "power"
raw_data_path = "data/processed_data/combined_data.csv"

def read_raw_data_and_sort(raw_data_path: str, sort_by_cols: list = ["device", "datetime"]) -> pd.DataFrame:
    df_raw_data = pd.read_csv(raw_data_path, parse_dates=["datetime"])
    # make sure the sorting is correct
    df_raw_data = df_raw_data.sort_values(by=sort_by_cols).reset_index(drop=True)
    return df_raw_data

def get_test_size(days):
    # since the data is one minute one row
    return days * 24 * 60


df_raw_data = read_raw_data_and_sort(raw_data_path, sort_by_cols=["device", "datetime"])

# parameterize the device for testing conveniently
df_device = choose_device(df_raw_data, device_name)

# preprocssing
df_preprocessing = preprocess(df_device, cols_for_drop, preprocess_config=None)

# feature engineering
df_fe_result = feature_engineering(df=df_preprocessing, fe_config=fe_config)

# standardization
df_standardization = standardization(df_fe_result)

## Train Using Cross Validation
tss = TimeSeriesSplit(n_splits=n_splits, test_size=get_test_size(2), gap=day_gap)

valid_scores, result_df = train_and_valid(df_standardization, tss, target, invalid_cols_for_training)
print(f"Total absolute error: {np.mean(valid_scores):.2f}")


if pred_result_ouput:

    output_dir = './pred_results/Sean'  # Output directory
    output_file = f"{output_dir}/{test_name}_result.csv"  # Output file path
    
    # Check if the file already exists
    if os.path.exists(output_file):
        user_input = input(f"The file '{output_file}' already exists. Do you want to replace it? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting without replacing the file.")
            exit(0)  # Stop the script
        else:
            print(f"Replacing the file '{output_file}'...")
    else:
        print(f"The file '{output_file}' does not exist. Proceeding to save results.")

    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result_df['pred_time'] = current_date
    result_df.to_csv(output_file, index=False)