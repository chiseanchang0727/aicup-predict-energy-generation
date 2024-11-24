
import pandas as pd
from sklearn.preprocessing import StandardScaler

def select_initial_columns(cols_for_drop):
    base_columns = ['windspeed', 'pressure', 'temperature', 'humidity', 'sunlight']
    initial_cols = [col for col in base_columns if col not in cols_for_drop]
    return initial_cols



def standardization(df):

    scaler = StandardScaler()
    df_standardized = df.copy()

    # Select numerical columns only
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Standardize the numerical columns
    df_standardized[numerical_cols] = scaler.fit_transform(df_standardized[numerical_cols])


    return df_standardized