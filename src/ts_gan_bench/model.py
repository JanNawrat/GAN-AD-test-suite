import torch
import torch.nn as nn

class LSTM_Generator(nn.Module):
    def __init__(
            self,
            in_dim=1,
            out_dim=1,
            hidden_size=100,
            num_layers=3
        ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, out_dim), nn.Tanh())

    def forward(self, z):
        sequence, (h_n, c_n) = self.lstm(z)
        projected_sequence = self.proj(sequence)
        return projected_sequence
    
class LSTM_Discriminator(nn.Module):
    def __init__(
            self, 
            in_dim=1,
            hidden_size=100,
            num_layers=1
        ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, (h_n, c_n) = self.lstm(x)
        projected_sequence = self.proj(sequence)
        return projected_sequence
