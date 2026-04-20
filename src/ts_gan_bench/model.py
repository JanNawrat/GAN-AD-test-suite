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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, out_dim), nn.Tanh())

    def forward(self, z):
        sequence, (h_n, c_n) = self.lstm(z)
        projected_sequence = self.proj(sequence)
        return projected_sequence
    
    def save(self, path):
        checkpoint = {
            'config': {
                'in_dim': self.in_dim,
                'out_dim': self.out_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
            },
            'weights': self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(
        cls,
        path,
        map_location,
    ):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint.pop('config'))
        model.load_state_dict(checkpoint.pop('weights'))
        return model
    
class LSTM_Discriminator(nn.Module):
    def __init__(
            self, 
            in_dim=1,
            hidden_size=100,
            num_layers=1
        ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, (h_n, c_n) = self.lstm(x)
        projected_sequence = self.proj(sequence)
        return projected_sequence
    
    def save(self, path):
        checkpoint = {
            'config': {
                'in_dim': self.in_dim,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
            },
            'weights': self.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(
        cls,
        path,
        map_location,
    ):
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint.pop('config'))
        model.load_state_dict(checkpoint.pop('weights'))
        return model
