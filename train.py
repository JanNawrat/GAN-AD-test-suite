import tomli_w
import torch

import argparse
import json
from pathlib import Path
import tomllib

from dataset import NASA_dataloader
from model import MAD_GAN_Generator, MAD_GAN_Discriminator
from trainer import ReverseMapTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SETTINGS_ROOT = Path('settings')
STATES_ROOT = Path('states')
DATA_ROOT = Path('data')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--settings', type=str, default='default')
    parser.add_argument('-n', '--n-epochs', type=int, default=1)
    parser.add_argument('--model-save-frequency', type=int, default=None)
    parser.add_argument('--use-mps', action='store_true')
    parser.add_argument('--overwrite', action='store_true') # used for debugging, will overwrite results
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # mps (apple silicon only)
    if args.use_mps and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')

    # importing settings
    with open(SETTINGS_ROOT / f'{args.settings}.toml', 'rb') as file:
        settings = tomllib.load(file)

    # loading models
    match settings['model']['generator']:
        case 'mad_gan':
            generator = MAD_GAN_Generator(settings['params']['frame_length'], settings['model']['zdim']).to(DEVICE)
        case _:
            print('No generator selected.')
            raise SystemExit(1)
        
    match settings['model']['discriminator']:
        case 'mad_gan':
            discriminator = MAD_GAN_Discriminator(settings['params']['frame_length']).to(DEVICE)
        case _:
            print('No discriminator selected.')
            raise SystemExit(1)
        
    # loading dataset
    match settings['dataset']['name']:
        case 'nasa':
            train_loader = NASA_dataloader(settings, DATA_ROOT, train=True)
        case _:
            print('No dataset selected.')
            raise SystemExit(1)
    
    # initializing trainer:
    match settings['model']['trainer']:
        case 'reverse_map':
            trainer = ReverseMapTrainer(settings, generator, discriminator, train_loader, DEVICE, STATES_ROOT / args.experiment_name)
        case _:
            print('No trainer selected.')
            raise SystemExit(1)
        
    # attempting to prepare the state directory
    state_dir = STATES_ROOT / args.experiment_name
    try:
        state_dir.mkdir(parents=False, exist_ok=args.overwrite)
    except:
        print(f'Couldn\'t create directory {state_dir}. Make sure the parent directory exists and name isn\'t already taken.')
        raise SystemExit(1)

    # starting training
    trainer.train(args.n_epochs, args.model_save_frequency)

    # saving settings
    with open(state_dir / 'settings.toml', 'wb') as f:
        tomli_w.dump(settings, f)
