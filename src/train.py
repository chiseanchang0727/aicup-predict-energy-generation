import xgboost as xgb

xgb_params = {
    "base_score": 0.1,
    "booster": "gbtree",
    "n_estimators": 1000,
    "early_stopping_rounds": 100,
    "max_depth": 10,
    "learning_rate": 1e-2,
}


def initialize_xgb_model(**kwargs):
    return xgb.XGBRegressor(**kwargs)

def train_and_valid(df, tss, target, drop_cols, xgb_params=xgb_params):
    
    print('Training starts.')
    preds = []
    valid_scores = []
    for train_idx, val_idx in tss.split(df):
        df_train, df_valid = df.iloc[train_idx], df.iloc[val_idx]

        X_train, y_train = df_train.drop(drop_cols, axis=1), df_train[target]
        X_valid, y_valid = df_valid.drop(drop_cols, axis=1), df_valid[target]

        reg = initialize_xgb_model(**xgb_params)
        reg.fit(
            X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False
        )

        y_pred = reg.predict(X_valid)
        preds.append(y_pred)
        score = sum(abs(y_pred - y_valid))
        valid_scores.append(score)
    
    return valid_scores

