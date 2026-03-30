import torch

import argparse
import json
from pathlib import Path

from model import MAD_GAN_Generator, MAD_GAN_Discriminator
from dataset import NASA_dataloader
from trainer import ReverseMapTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SETTINGS_ROOT = Path('settings/model')
DATASET_SETTINGS_ROOT = Path('settings/dataset')
STATES_ROOT = Path('states')
DATA_ROOT = Path('data')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--model-settings', type=str, default='default')
    parser.add_argument('--dataset-settings', type=str, default='default')
    parser.add_argument('--n-epochs', type=int, default=1)
    parser.add_argument('--model-save-frequency', type=int, default=None)
    parser.add_argument('--use-mps', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # mps (apple silicon only)
    if args.use_mps and torch.backends.mps.is_available():
        DEVICE = torch.device('mps')

    # importing settings
    with open(MODEL_SETTINGS_ROOT / f'{args.model_settings}.json', 'r') as file:
        model_settings = json.load(file)

    with open(DATASET_SETTINGS_ROOT / f'{args.dataset_settings}.json', 'r') as file:
        dataset_settings = json.load(file)

    # loading models
    match model_settings['generator']:
        case 'mad_gan':
            generator = MAD_GAN_Generator(model_settings['frame_length'], model_settings['zdim'])
        case _:
            print('No generator selected.')
            raise SystemExit(1)
        
    match model_settings['discriminator']:
        case 'mad_gan':
            discriminator = MAD_GAN_Discriminator(model_settings['frame_length'])
        case _:
            print('No discriminator selected.')
            raise SystemExit(1)
        
    # loading dataset
    match dataset_settings['dataset']:
        case 'nasa':
            train_loader = NASA_dataloader(dataset_settings, DATA_ROOT, train=True)
        case _:
            print('No dataset selected.')
            raise SystemExit(1)
    
    # initializing trainer:
    match model_settings['trainer']:
        case 'reverse_map':
            trainer = ReverseMapTrainer(model_settings, generator, discriminator, train_loader, STATES_ROOT / args.experiment_name)
        case _:
            print('No trainer selected.')
            raise SystemExit(1)
        
    # attempting to prepare the state directory
    state_dir = STATES_ROOT / args.experiment_name
    try:
        state_dir.mkdir(parents=False, exist_ok=False)
    except:
        print(f'Couldn\'t create directory {state_dir}. Make sure the parent directory exists and name isn\'t already taken.')
        raise SystemExit(1)

    # starting training
    trainer.train(args.n_epochs, args.model_save_frequency)

    # saving settings
    with open(state_dir / 'model_settings.json', 'w') as f:
        json.dump(model_settings, f)
    
    with open(state_dir / 'dataset_settings.json', 'w') as f:
        json.dump(dataset_settings, f)
