import os
import pandas as pd


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
    return df[df['device'] == device]