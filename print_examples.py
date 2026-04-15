# ==========================================
# Generates example outputs of trained model
# Currently supports SWaT
# ==========================================

import matplotlib.pyplot as plt
import tomli_w
import torch

import argparse
import json
from pathlib import Path
import time
import tomllib

from consts import DEVICE, SETTINGS_ROOT, STATES_ROOT, DATA_ROOT
from dataset import NASA_dataloader, SWaT_dataloader, get_SWaT_column_names
from model import LSTM_Generator, LSTM_Discriminator
from trainer import ReverseMapTrainer

# parse args
# load settings (in states)
# load model
# select training type and generate examples
# save examples

BATCH_SIZE = 8


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('-n', type=int, default=-1) # load model snapshot at selected epoch
    parser.add_argument('--use-mps', action='store_true')
    parser.add_argument('--overwrite', action='store_true') # used for debugging, will overwrite results
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # mps (apple silicon only)
    if args.use_mps and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    print(f'Selected device: {DEVICE}')

    # importing settings
    with open(STATES_ROOT / args.experiment_name / 'settings.toml', 'rb') as file:
        settings = tomllib.load(file)

    # loading models
    match settings['model']['generator']['name']:
        case 'lstm':
            generator = LSTM_Generator(
                settings['model']['generator']['in_dim'],
                settings['model']['generator']['out_dim'],
                settings['model']['generator']['hidden_size'],
                settings['model']['generator']['num_layers'],
            ).to(DEVICE)
        case _:
            print('No generator selected.')
            raise SystemExit(1)
    
    # loading state dict
    checkpoint = torch.load(
        STATES_ROOT / args.experiment_name / 'generator' / f'G_{args.n}.pth',
        weights_only=True,
        map_location=DEVICE,
    )
    generator.load_state_dict(checkpoint)
    
    # generating examples:
    match settings['model']['trainer']:
        case 'reverse_map':
            frame_length = settings['params']['frame_length']
            zdim = settings['model']['generator']['in_dim']
            z = torch.randn(BATCH_SIZE, frame_length, zdim, device=DEVICE)
            fake_sequences = generator(z).detach().cpu().numpy()
        case _:
            print('No trainer selected.')
            raise SystemExit(1)
        
    # loading sensor names
    column_names = get_SWaT_column_names(settings, DATA_ROOT)

    # generaing plot
    fig = plt.figure(layout='constrained', figsize=(16, 100))
    subfigs = fig.subfigures(fake_sequences.shape[2])
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(column_names[i], fontsize='x-large', fontweight='bold')
        axs = subfig.subplots(1, BATCH_SIZE)
        for j, ax in enumerate(axs):
            ax.plot(fake_sequences[j,:,i])

    # saving plot
    examples_dir = STATES_ROOT / args.experiment_name / 'examples'
    examples_dir.mkdir(parents=False, exist_ok=True)
    plt.savefig(examples_dir / f'example_{args.n}.png')