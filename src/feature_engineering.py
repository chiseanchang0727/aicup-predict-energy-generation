import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from src.fe_tools import (
    create_time_features,
    create_sinusoidal_transformation_by_month,
    create_pe
)



def create_sinusoidal_date_base(period):

    # create sinusodial month mapping
    if period == 'date':
        df_base = pd.date_range(start='2023-01-01', end='2023-12-31')
        
    elif period == 'month':
        month_numbers = list(range(1, 13))
        df_base = pd.DataFrame(month_numbers, columns=['month'])
        
    # lunar_dates_params = {
    #     "start_date": "2023-02-04", 
    #     "end_date": "2024-02-03",
    #     "col_name": "lunar_ymd"
    # }
    
    df_pe = create_sinusoidal_transformation_by_month(df_base, 12)
    # df_pe = create_pe(df_base, d=4, n=1e3)
    return df_pe

def positional_encoding(df, period):
    
    df_fe_result = None
    if period == 'date':
        df_pe = create_sinusoidal_date_base(period)
        df = create_time_features(df, "datetime")
        df_fe_result = pd.merge(
            df, df_pe, how="left", on=["year", "month", "day"]
        )
    elif period == 'month':
        df_pe = create_sinusoidal_date_base(period)
         # Aligns with a specific lunar calendar period
        df = create_time_features(df, "datetime")
        df_fe_result = pd.merge(
            df, df_pe, how="left", on=["month"]
        )   

    else:
        raise ValueError('Non accepted positional encoding method.')
    
    return df_fe_result



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

def drop_date_columns(df):

    columns_to_drop = [col for col in df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
    
    df = df.drop(columns=columns_to_drop)
    
    return df


def feature_engineering(df, fe_config):
    # df = calculate_pressure_diff(df, column="pressure")

    pe_config = fe_config['pe_config']
    pe_flag = pe_config['flag']
    pe_period = pe_config['period']

    if pe_flag:
        df = positional_encoding(df, period=pe_period).copy()
    else:
        df = df.copy()


    sunlight_sim_config = fe_config['sunlight_sim_config']
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
    else:
        df_fe_result = df

    return df_fe_result