import xgboost as xgb

xgb_params = {
    "base_score": 0.5,
    "booster": "gbtree",
    "n_estimators": 100,
    "early_stopping_rounds": 50,
    "max_depth": 3,
    "learning_rate": 1e-2,
}

def initialize_xgb_model(**kwargs):
    return xgb.XGBRegressor(**kwargs)

def generate_scores_by_xgb(df, tss, TARGET="power", drop_cols=["power", "device", "date"], xgb_params=xgb_params):
    preds = []
    scores = []
    for train_idx, val_idx in tss.split(df):
        df_train, df_valid = df.iloc[train_idx], df.iloc[val_idx]

        X_train, y_train = df_train.drop(drop_cols, axis=1), df_train[TARGET]
        X_valid, y_valid = df_valid.drop(drop_cols, axis=1), df_valid[TARGET]

        reg = initialize_xgb_model(**xgb_params)
        reg.fit(
            X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=False
        )

        y_pred = reg.predict(X_valid)
        preds.append(y_pred)
        score = sum(abs(y_pred - y_valid))
        scores.append(score)
    
    return scores

