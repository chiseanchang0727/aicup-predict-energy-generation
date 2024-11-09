import pandas as pd


# create time related features
def create_time_features(df, input_column) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[input_column] = pd.to_datetime(df_copy[input_column])
    df_copy['date'] = df_copy[input_column].dt.date
    df_copy['year'] = df_copy[input_column].dt.year
    df_copy['month'] = df_copy[input_column].dt.month
    df_copy['hour'] = df_copy[input_column].dt.hour
    df_copy['min'] = df_copy[input_column].dt.minute

    return df_copy