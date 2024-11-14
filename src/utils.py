import os
import pandas as pd
import json

def load_dataframes(folder_path) -> pd.DataFrame:
    dataframes = pd.DataFrame()
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  
            device_name = filename.split('_')[0]
            df_name = "df_" + device_name
            df = pd.read_csv(os.path.join(folder_path, filename))
            df['device'] = device_name
            
            dataframes = pd.concat([dataframes, df], axis=0)

    return dataframes



def choose_device(df, device):
    df_copy = df[df['device'] == device].copy()
    df_copy = df_copy.drop('locationcode', axis=1)
    df_copy = df_copy.set_index('datetime')
    return df_copy



def read_config(path):
    try:
        with open(path, 'r') as file:
            configs = json.load(file)

        return configs
    except FileNotFoundError:
        print(f"The file {path} was not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {path}.")