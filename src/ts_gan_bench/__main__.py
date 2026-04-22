from argparse import ArgumentParser
import shutil
import time

import torch

from ts_gan_bench import constants
from ts_gan_bench.dataloader import SWaT_dataloader
from ts_gan_bench.model import LSTM_Generator, LSTM_Discriminator
from ts_gan_bench.settings import load_settings
from ts_gan_bench.trainer import ReverseMapTrainer

def parse_arguments():
    parser = ArgumentParser()
    # # device args
    # device_group = parser.add_mutually_exclusive_group()
    # device_group.add_argument('--cpu', action='store_const', const='cpu', dest='device', help='Run models on CPU')
    # device_group.add_argument('--gpu', action='store_const', const='gpu', dest='device', help='Run models on GPU')
    # device_group.add_argument('--mps', action='store_const', const='gpu', dest='device', help='Run models on MPS')
    # subparsers
    subparsers = parser.add_subparsers(dest='mode', required=True)
    # train args
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('experiment_name', type=str)
    train_parser.add_argument('-n', '--n_epochs', type=int, default=1)
    train_parser.add_argument('--settings', type=str, default='default')
    train_parser.add_argument('--save-freq', type=int, default=None)
    train_parser.add_argument('--overwrite', action='store_true') # used for debugging, will overwrite results
    # test args
    test_parser = subparsers.add_parser('test')
    # TODO: test

    return parser.parse_args()

def load_generator(generator_settings):
    generator_type = generator_settings.type
    generator_kwargs = generator_settings.model_dump(exclude={'type'})
    match generator_type:
        case 'lstm':
            return LSTM_Generator(**generator_kwargs)
        case _:
            print('Incorrect generator selected!')
            raise SystemExit(1)
        
def load_discriminator(discriminator_settings):
    discriminator_type = discriminator_settings.type
    discriminator_kwargs = discriminator_settings.model_dump(exclude={'type'})
    match discriminator_type:
        case 'lstm':
            return LSTM_Discriminator(**discriminator_kwargs)
        case _:
            print('Incorrect discriminator selected!')
            raise SystemExit(1)
        
def load_dataset(settings): # temporary, TODO: improve
    return SWaT_dataloader(settings)

def main():
    args = parse_arguments()
    settings = load_settings(constants.SETTINGS_ROOT / f'{args.settings}.toml', args.experiment_name, args.n_epochs)
    # print(settings.model_dump_json(indent=4))

    # load models
    generator = load_generator(settings.model.generator).to(torch.device(settings.device_name))
    discriminator = load_discriminator(settings.model.discriminator).to(torch.device(settings.device_name))
    
    # load dataset
    train_loader, _, _, actuator_idx = load_dataset(settings)

    # initialize trainer
    match settings.model.type:
        case 'reverse_map':
            trainer = ReverseMapTrainer(
                settings=settings,
                generator=generator,
                discriminator=discriminator,
                train_loader=train_loader,
            )
        case _:
            print('Incorrect model type!')
            raise SystemExit(1)
    trainer.add_actuator_idx(actuator_idx)
        
    # attempting to prepare the state directory
    state_dir = settings.paths.state_dir
    try:
        state_dir.mkdir(parents=False, exist_ok=args.overwrite)
    except:
        print(f'Couldn\'t create directory {state_dir}. Make sure the parent directory exists and name isn\'t already taken.')
        raise SystemExit(1)
    
    # saving setting (only the original .toml file)
    shutil.copyfile(
        constants.SETTINGS_ROOT / f'{args.settings}.toml',
        settings.paths.state_dir / 'settings.toml'
    )
    
    # starting training
    time_start = time.time()
    trainer.train(args.n_epochs, args.save_freq)
    time_end = time.time()
    time_total = time_end - time_start
    time_per_epoch = time_total / args.n_epochs
    print(f'Total training time: {time.strftime('%H:%M:%S', time.gmtime(time_total))}')
    print(f'Time per epoch: {time.strftime('%M:%S', time.gmtime(time_per_epoch))}')

if __name__ == '__main__':
    main()
