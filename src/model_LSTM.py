import torch.nn as nn


class LSTMmodel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers,  dropout=0.2, device="cpu"):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout).to(device)
        self.fc = nn.Linear(hidden_size, output_size).to(device)
        
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x