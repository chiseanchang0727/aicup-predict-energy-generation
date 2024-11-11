import pandas as pd
import numpy as np

def calculate_pressure_diff(df, column='pressure'):
    """
    Adds a new column to the DataFrame with the difference of each value
    in the specified column from the column's mean.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The column for which to calculate the difference from the mean.

    Returns:
    pd.DataFrame: The DataFrame with an added column for pressure difference.
    """
    df_copy = df.copy()
    mean_value = df_copy[column].mean()
    df_copy.loc[:, f'{column}_diff'] = df_copy[column].apply(lambda x: round(x - mean_value, 2))
    return df_copy

# create time related features
def create_time_features(df, input_column) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[input_column] = pd.to_datetime(df_copy[input_column])
    df_copy['date'] = df_copy[input_column].dt.date
    df_copy['year'] = df_copy[input_column].dt.year
    df_copy['month'] = df_copy[input_column].dt.month
    df_copy['day'] = df_copy[input_column].dt.day
    df_copy['hour'] = df_copy[input_column].dt.hour
    df_copy['min'] = df_copy[input_column].dt.minute

    # Seasonal features
    df_copy['day_of_week'] = df_copy[input_column].dt.dayofweek  # Monday=0, Sunday=6
    df_copy['week_of_year'] = df_copy[input_column].dt.isocalendar().week  # Week of the year
    df_copy['quarter'] = df_copy[input_column].dt.quarter 
    
    # df_copy['season'] = df_copy['month'].apply(lambda x: 
    #                                            'winter' if x in [12, 1, 2] else
    #                                            'spring' if x in [3, 4, 5] else
    #                                            'summer' if x in [6, 7, 8] else
    #                                            'fall')

    return df_copy


def create_sinusoidal_transformation_by_number(df, col_name, period):
    """
    Adds sinusoidal transformation columns (sin and cos) for a given column.
    
    Parameters:
    - df: DataFrame to add the transformations to.
    - col_name: The column on which to perform the transformations.
    - period: The period of the cycle (e.g., 12 for months, 7 for days of the week).
    
    Returns:
    - DataFrame with added columns for sinusoidal transformations.
    """
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / period)
    return df



def create_sinusoidal_transformation_year_month_day(df, col_name, year, month, day, period):
    """
    Adds sinusoidal transformation columns (sin and cos) for year, month, day.
    
    Parameters:
    - df: DataFrame to add the transformations to.
    - year: the column stands for year..
    - month: the column stands for month.
    - day: the column stands for day.
    - period: The period of the cycle (e.g., 12 for months, 7 for days of the week).
    
    Returns:
    - DataFrame with added columns for sinusoidal transformations.
    """
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[year] * df[month] * df[day] / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[year] * df[month] * df[day] / period)
    return df