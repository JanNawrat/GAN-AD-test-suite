import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETTINGS_ROOT = Path('settings')
STATES_ROOT = Path('states')
DATA_ROOT = Path('data')