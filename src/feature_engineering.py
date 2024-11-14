import pandas as pd


from src.fe_tools import (
    create_time_features,
    create_sinusoidal_transformation_by_month,
    create_pe
)



def create_sinusoidal_date_base(period):

    # create sinusodial month mapping
    if period == 'date':
        df_base = pd.date_range(start="2023-01-01", end="2023-12-31")
        
    elif period == 'month':
        month_numbers = list(range(1, 13))
        df_base = pd.DataFrame(month_numbers, columns=["month"])
        
    # lunar_dates_params = {
    #     "start_date": "2023-02-04", 
    #     "end_date": "2024-02-03",
    #     "col_name": "lunar_ymd"
    # }
    
    df_pe = create_sinusoidal_transformation_by_month(df_base, 'month', 12)
    df_pe = create_pe(df_base, d=4, n=1e3)
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

def feature_engineering(df, pe_config):
    # df = calculate_pressure_diff(df, column="pressure")

    pe_flag = pe_config['flag']
    pe_period = pe_config['period']
    if pe_flag:
        df_fe_result = positional_encoding(df, period=pe_period)
    else:
        df_fe_result = df

    df_fe_result = drop_date_columns(df_fe_result)
    return df_fe_result



def drop_date_columns(df):

    columns_to_drop = [col for col in df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
    
    df = df.drop(columns=columns_to_drop)
    
    return df
