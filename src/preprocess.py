from sklearn.preprocessing import StandardScaler




def preprocess(df, columns_to_standardize, cols_for_drop):

    df_2 = col_remove(df, cols_for_drop)

    scaler = StandardScaler()
    df_standardized = df_2.copy()
    df_standardized[columns_to_standardize] = scaler.fit_transform(
        df_standardized[columns_to_standardize]
    )
    return df_standardized




def col_remove(df, cols_for_drop):

    df_result= df.copy()
    if cols_for_drop:
        df_result = df.drop(cols_for_drop, axis=1).copy()

    return df_result