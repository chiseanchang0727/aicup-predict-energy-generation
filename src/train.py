import xgboost as xgb
import pandas as pd

xgb_params = {
    "base_score": 0.1,
    "booster": "gbtree",
    "n_estimators": 1000,
    "early_stopping_rounds": 100,
    "max_depth": 10,
    "learning_rate": 1e-2,
}

def drop_invalid_columns(df, invalid_cols):

    columns_to_drop = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in invalid_cols)]
    
    df = df.drop(columns=columns_to_drop)
    
    return df

def initialize_xgb_model(**kwargs):
    return xgb.XGBRegressor(**kwargs)

def train_and_valid(df, tss, target, invalid_cols, xgb_params=xgb_params):
    # df = drop_invalid_columns(df, invalid_cols)
    print('Training starts.')

    result_df = pd.DataFrame()
    valid_scores = []
    fold = 0
    for train_idx, val_idx in tss.split(df):
        df_train, df_valid = df.iloc[train_idx], df.iloc[val_idx]

        X_train, y_train = df_train.drop(target, axis=1), df_train[target]
        X_valid, y_valid = df_valid.drop(target, axis=1), df_valid[target]

        X_train = drop_invalid_columns(X_train, invalid_cols)
        X_valid = drop_invalid_columns(X_valid, invalid_cols)

        reg = initialize_xgb_model(**xgb_params)
        reg.fit(
            X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False
        )

        y_pred = reg.predict(X_valid)


        # Store y_pred, y_valid, and datetime index for each fold in a temporary DataFrame
        df_pred_valid = pd.DataFrame({
            'fold': fold,
            'datetime': df_valid.datetime,  # Add datetime index
            'y_valid': y_valid,
            'y_pred': y_pred
        })

        df_pred_valid['datetime'] = pd.to_datetime(df_pred_valid['datetime'])
        df_pred_valid['tae'] = abs(df_pred_valid['y_valid'] - df_pred_valid['y_pred'])

        scores = df_pred_valid['tae'].sum()
        valid_scores.append(scores)

        result_df = pd.concat([result_df, df_pred_valid], ignore_index=True)
        fold += 1

    return valid_scores, result_df

