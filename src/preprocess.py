import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter

def replace_saturated_sunlight(df, window_lenght, polyorder, saturation_value=117758.2):
    smoothed_data = df.copy()

    smoothed_data['sunlight'] = savgol_filter(
        smoothed_data['sunlight'].where(smoothed_data['sunlight'] < saturation_value, np.nan).fillna(saturation_value),
        window_length=window_lenght,
        polyorder=polyorder
    )

    df_sim_sunlight = smoothed_data[['sunlight']].rename(columns={'sunlight':'sim_sunlight'})
    df_merge = pd.merge(df, df_sim_sunlight, how='inner', on=df.index)
    df_merge['sunlight'] = np.where((df_merge['sunlight'] == saturation_value) & (df_merge['sim_sunlight'] > df_merge['sunlight']), 
                                    df_merge['sim_sunlight'], df_merge['sunlight'])
    df_merge = df_merge.drop(['sim_sunlight'], axis=1)
    # df_merge = df_merge.set_index('key_0')
    return df_merge

def select_initial_columns(cols_for_drop):
    base_columns = ['windspeed', 'pressure', 'temperature', 'humidity', 'sunlight']
    initial_cols = [col for col in base_columns if col not in cols_for_drop]
    return initial_cols

def col_remove(df, cols_for_drop):

    df_result= df.copy()
    if cols_for_drop:
        df_result = df.drop(cols_for_drop, axis=1).copy()

    return df_result

def preprocess(df, cols_for_drop, preprocess_config):

    sunlight_sim_config = preprocess_config['sunlight_sim_config']
    if sunlight_sim_config['flag']:
        window_length = sunlight_sim_config['window_length']
        polyorder = sunlight_sim_config['polyorder']

        df['date'] = pd.to_datetime(df['datetime']).dt.date

        df_sunsim_result = pd.DataFrame()
        for date in df['date'].unique():
            try:
                temp = replace_saturated_sunlight(df[df['date'] == date], window_lenght=window_length, polyorder=polyorder)
            
            except:
                continue

            df_sunsim_result = pd.concat([df_sunsim_result, temp], axis=0)

    df_sunsim_result = df_sunsim_result.drop(['date','key_0'], axis=1)
    initial_cols = select_initial_columns(cols_for_drop)

    df_2 = col_remove(df_sunsim_result, cols_for_drop)

    scaler = StandardScaler()
    df_standardized = df_2.copy()
    df_standardized[initial_cols] = scaler.fit_transform(
        df_standardized[initial_cols]
    )
    return df_standardized


