from sklearn.preprocessing import StandardScaler

def preprocess(df_device, columns_to_standardize):
    scaler = StandardScaler()
    df_standardized = df_device.copy()
    df_standardized[columns_to_standardize] = scaler.fit_transform(
        df_standardized[columns_to_standardize]
    )
    return df_standardized