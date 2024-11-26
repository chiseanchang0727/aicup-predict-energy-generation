import torch.nn as nn


class LSTMmodel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,  dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x