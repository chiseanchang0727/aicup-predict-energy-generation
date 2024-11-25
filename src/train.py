import xgboost as xgb
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model_LSTM import LSTMmodel

xgb_params = {
    "base_score": 0.1,
    "booster": "gbtree",
    "n_estimators": 1000,
    "early_stopping_rounds": 100,
    "max_depth": 10,
    "learning_rate": 1e-2,
}

lstm_params = {
    "batch_size": 10,
    "n_epochs": 10,
    
}

def get_model(input_size, hidden_size,  num_layers, output_size, dropout):
    
    model = LSTMmodel(input_size, hidden_size, num_layers, output_size, dropout)
    
    return model



def drop_invalid_columns(df, invalid_cols):

    columns_to_drop = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in invalid_cols)]
    
    df = df.drop(columns=columns_to_drop)
    
    return df


def create_loader(X: pd.DataFrame, y: pd.DataFrame, batch_size):
    dataset = TensorDataset(
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32)
    )
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    return dataloader

def train_model(model, dataloader, criterion, optimzier, device):
    """
    Note that the criterion is used for evaluating the performance by counting the error.
    It's usually by like nn.MSELoss(), but we use homemade function at first, it should be the same.
    """
    model.train()
    total_loss = 0.0
    
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # initial optimizer
        optimzier.zero_grad()    
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimzier.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def initialize_xgb_model(**kwargs):
    return xgb.XGBRegressor(**kwargs)

def train_and_valid(df, tss, target, invalid_cols, batch_size, hyperparams):
    # df = drop_invalid_columns(df, invalid_cols)
    print('Training starts.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    result_df = pd.DataFrame()
    valid_scores = []
    fold = 0
    n_epochs = 10
    for train_idx, val_idx in tss.split(df):
        df_train, df_valid = df.iloc[train_idx], df.iloc[val_idx]

        X_train, y_train = df_train.drop(target, axis=1), df_train[target]
        X_valid, y_valid = df_valid.drop(target, axis=1), df_valid[target]
        
        # index for error analysis
        result_index = df_valid.datetime

        X_train = drop_invalid_columns(X_train, invalid_cols)
        X_valid = drop_invalid_columns(X_valid, invalid_cols)

        #TODO: reshape X_train, X_valid

        # Convert data to PyTorch tensors
        train_loader = create_loader(X_train, y_train, batch_size)
        valid_loader = create_loader(X_valid, y_valid, batch_size)

        # Initial model
        model = get_model(input_size, hidden_size, **hyperparams)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        best_loss = float('inf')
        for epoch in n_epochs:
            train_loss = train_model(model, train_loader, criterion, optimizer, device)





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

