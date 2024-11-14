from sklearn.preprocessing import StandardScaler


def select_initial_columns(cols_for_drop):
    base_columns = ['windspeed', 'pressure', 'temperature', 'humidity', 'sunlight']
    initial_cols = [col for col in base_columns if col not in cols_for_drop]
    return initial_cols

def col_remove(df, cols_for_drop):

    df_result= df.copy()
    if cols_for_drop:
        df_result = df.drop(cols_for_drop, axis=1).copy()

    return df_result

def preprocess(df, cols_for_drop):
    initial_cols = select_initial_columns(cols_for_drop)

    df_2 = col_remove(df, cols_for_drop)

    scaler = StandardScaler()
    df_standardized = df_2.copy()
    df_standardized[initial_cols] = scaler.fit_transform(
        df_standardized[initial_cols]
    )
    return df_standardized


