import pandas as pd


from fe_tools import (
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


## Feature engineering


def feature_engineering(df):
    df_fe_result = calculate_pressure_diff(df, column="pressure")

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




    return df_fe_result