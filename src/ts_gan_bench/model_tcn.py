import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        dilation,
        dropout=0.2,
        use_spectral_norm=False,
    ):
        padding = (kernel_size - 1) * dilation
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_dim,
            out_dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if use_spectral_norm:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
        else:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    
class TCN_Generator(nn.Module):
    def __init__(
        self,
        in_dim=15,
        out_dim=50,
        kernel_size=3,
        num_channels=[128, 128, 128, 128],
        dilations=[1, 2, 4, 8],
        dropout=0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dilations = dilations
        self.dropout = dropout
        layers = []
        for i in range(len(num_channels)):
            layers.append(TemporalBlock(
                in_dim=in_dim if i == 0 else num_channels[i-1],
                out_dim=num_channels[i],
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout,
                use_spectral_norm=False,
            ))
        self.tcn = nn.Sequential(*layers)
        self.final_conv = weight_norm(nn.Conv1d(
            in_channels=num_channels[len(num_channels)-1],
            out_channels=out_dim,
            kernel_size=1,
        ))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tcn(x)
        x = self.final_conv(x)
        return self.tanh(x)
    
    def save(self, path):
        checkpoint = {
            'config': {
                'in_dim': self.in_dim,
                'out_dim': self.out_dim,
                'kernel_size': self.kernel_size,
                'num_channels': self.num_channels,
                'dialtions': self.dilations,
                'dropout': self.dropout,
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
        
class TCN_Discriminator(nn.Module):
    def __init__(
        self,
        in_dim=50,
        kernel_size=3,
        num_channels=[64, 64, 64],
        dilations=[1, 4, 16],
        dropout=0.2,      
    ):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.dilations = dilations
        self.dropout = dropout
        layers = []
        for i in range(len(num_channels)):
            layers.append(TemporalBlock(
                in_dim=in_dim if i == 0 else num_channels[i-1],
                out_dim=num_channels[i],
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout,
                use_spectral_norm=True,
            ))
        self.tcn = nn.Sequential(*layers)
        self.fc = spectral_norm(nn.Linear(num_channels[len(num_channels)-1], 1))

    def forward(self, x):
        x = self.tcn(x)
        x = torch.mean(x, dim=-1)
        return self.fc(x)
    
    def save(self, path):
        checkpoint = {
            'config': {
                'in_dim': self.in_dim,
                'kernel_size': self.kernel_size,
                'num_channels': self.num_channels,
                'dilations': self.dilations,
                'dropout': self.dropout,
            },
            'weight': self.state_dict(),
        }
        torch.save(checkpoint, path)
