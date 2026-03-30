import torch
import torch.nn as nn

class MAD_GAN_Generator(nn.Module):
    def __init__(self, seq_length, in_dim=1, hidden_size=100, num_layers=3, out_dim=1):
        super().__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, out_dim), nn.Tanh())

    def forward(self, z):
        sequence, (h_n, c_n) = self.lstm(z)
        projected_sequence = self.proj(sequence)
        return projected_sequence
    
class MAD_GAN_Discriminator(nn.Module):
    def __init__(self, seq_length, in_dim=1, hidden_size=100, num_layers=1):
        super().__init__()
        self.see_length = seq_length
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, x):
        sequence, (h_n, c_n) = self.lstm(x)
        projected_sequence = self.proj(sequence)
        return projected_sequence