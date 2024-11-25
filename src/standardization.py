
import numpy as np
from sklearn.preprocessing import StandardScaler

def select_initial_columns(cols_for_drop):
    base_columns = ['windspeed', 'pressure', 'temperature', 'humidity', 'sunlight']
    initial_cols = [col for col in base_columns if col not in cols_for_drop]
    return initial_cols



def standardization(df, log_trans_cols=None):

    # if log_trans_cols:
    #     df_transformed = df.copy()
    #     cols_to_transform = [col for col in df.columns if any(substr in col for substr in log_trans_cols)]
    #     for col in cols_to_transform:
    #         if col in df_transformed.columns:
    #             # Handle non-positive values by shifting (if needed)
    #             min_val = df_transformed[col].min()
    #             if min_val <= 0:
    #                 shift = abs(min_val) + 1  # Shift by the minimum value + 1 to avoid log(0)
    #                 print(f"Shifting column '{col}' by {shift} to handle non-positive values.")
    #                 df_transformed[col] = np.log(df_transformed[col] + shift)
    #             else:
    #                 df_transformed[col] = np.log(df_transformed[col])

    scaler = StandardScaler()
    df_standardized = df.copy()

    # Select numerical columns only
    numerical_cols = df_standardized.select_dtypes(include=['number']).columns

    # Standardize the numerical columns
    df_standardized[numerical_cols] = scaler.fit_transform(df_standardized[numerical_cols])


    return df_standardized