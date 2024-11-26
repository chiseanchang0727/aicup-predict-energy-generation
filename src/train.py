import xgboost as xgb
import pandas as pd
import numpy as np
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

def get_model(device, input_size,  output_size, hidden_size,  num_layers, dropout):
    
    model = LSTMmodel(input_size,  output_size, hidden_size, num_layers, dropout, device)
    
    return model.to(device)



def drop_invalid_columns(df, invalid_cols):

    columns_to_drop = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in invalid_cols)]
    
    df = df.drop(columns=columns_to_drop)
    
    return df


def create_loader(X: torch.tensor, y: torch.tensor, batch_size):
    dataset = TensorDataset(X, y)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    return dataloader

def train_model(epoch, model, dataloader, criterion, optimzier, device):
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
        # if epoch % 5 == 0:
        #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def valid_model(model, data_loader, criterion, device):

    model.eval()
    total_loss = 0.0
    predictions, actuals, = [], []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    return total_loss / len(data_loader), np.concatenate(predictions), np.concatenate(actuals)

def initialize_xgb_model(**kwargs):
    return xgb.XGBRegressor(**kwargs)

def train_and_valid(df, tss, target, invalid_cols, hyperparams):
    # df = drop_invalid_columns(df, invalid_cols)
    print('Training starts.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    result_df = pd.DataFrame()
    valid_scores = []
    fold = 0
    batch_size = 1024
    for train_idx, val_idx in tss.split(df):
        df_train, df_valid = df.iloc[train_idx], df.iloc[val_idx]

        X_train, y_train = df_train.drop(target, axis=1), df_train[target]
        X_valid, y_valid = df_valid.drop(target, axis=1), df_valid[target]
        

        X_train = drop_invalid_columns(X_train, invalid_cols)
        X_valid = drop_invalid_columns(X_valid, invalid_cols)

        # Convert to tensor
        X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train = torch.tensor(y_train.to_numpy().reshape(-1, 1), dtype=torch.float32)
        X_valid = torch.tensor(X_valid.to_numpy(), dtype=torch.float32)
        y_valid = torch.tensor(y_valid.to_numpy().reshape(-1, 1), dtype=torch.float32)

        # Convert data to PyTorch tensors
        train_loader = create_loader(X_train, y_train, batch_size)


        # Initial model
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        model = get_model(device, input_size, output_size, **hyperparams)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        epochs = 150

        total_step = len(train_loader)
        for epoch in range(epochs):
            # train_loss = train_model(epoch, model, train_loader, criterion, optimizer, device)
            # valid_loss, y_pred, y_true = valid_model(model, valid_loader, criterion, device)
            model.train()
            total_loss = 0.0
            
            for i, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # initial optimizer
                optimizer.zero_grad()    
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                # total_loss += loss.item() 

                if i % 10 == 0:
                    print(f'Fold {fold}, Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

        model.eval()
        with torch.no_grad():
            y_pred = model(X_valid.to(device))
            
        # # Store y_pred, y_valid, and datetime index for each fold in a temporary DataFrame
            df_pred_valid = pd.DataFrame({
                'fold': fold,
                'datetime': df_valid.datetime,  # Add datetime index
                'y_valid': y_valid.cpu().numpy().squeeze(),
                'y_pred': y_pred.cpu().numpy().squeeze()
            })

            df_pred_valid['datetime'] = pd.to_datetime(df_pred_valid['datetime'])
            df_pred_valid['tae'] = abs(df_pred_valid['y_valid'] - df_pred_valid['y_pred'])

            scores = df_pred_valid['tae'].sum()
            valid_scores.append(scores)

            result_df = pd.concat([result_df, df_pred_valid], ignore_index=True)
            
        fold += 1

    return valid_scores, result_df

