import pandas as pd


from src.fe_tools import (
    calculate_pressure_diff,
    create_sinusoidal_transformation_by_number,
    create_sinusoidal_transformation_year_month_day,
    create_time_features,
)


def generate_sinusoidal_date_features(start_date, end_date, col_name):
    dates = pd.date_range(start=start_date, end=end_date)

    # Create DataFrame and extract year, month, and day into separate columns
    df_dates = pd.DataFrame({
        "year": dates.year, 
        "month": dates.month, 
        "day": dates.day
    })

    return create_sinusoidal_transformation_year_month_day(df_dates, col_name, "year", "month", "day", 12)

def create_sinusoidal_date_base():

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


    return df_general_dates, df_lunar_dates

def positional_encoding(df, method):
    
    df_fe_result = None
    if method == 'general':
        df_general_dates, _ = create_sinusoidal_date_base()
        df = create_time_features(df, "datetime")
        df_fe_result = pd.merge(
            df, df_general_dates, how="left", on=["year", "month", "day"]
        )
    elif method == 'lunar':
        _, df_lunar_dates = create_sinusoidal_date_base()
         # Aligns with a specific lunar calendar period
        df = create_time_features(df, "datetime")
        df_fe_result = pd.merge(
            df, df_lunar_dates, how="left", on=["year", "month", "day"]
        )   

    else:
        raise ValueError('Non accepted positional encoding method.')
    
    return df_fe_result

def feature_engineering(df, pe_config):
    # df = calculate_pressure_diff(df, column="pressure")

    pe_flag = pe_config['flag']
    pe_method = pe_config['method']
    if pe_flag:
        df_fe_result = positional_encoding(df, method=pe_method)
    else:
        df_fe_result = df

    df_fe_result = drop_date_columns(df_fe_result)
    return df_fe_result



def drop_date_columns(df):

    columns_to_drop = [col for col in df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
    
    df = df.drop(columns=columns_to_drop)
    
    return df
