import tomllib
from typing import Literal, Annotated, Union
from pathlib import Path

from pydantic import BaseModel, Field
import torch

from ts_gan_bench import constants

class Paths(BaseModel):
    data_root: Path
    state_dir: Path

# =======
# Dataset
# =======
class SWaTConfig(BaseModel):
    type: Literal['swat']
    features: list

DatasetConfig = Annotated[
    Union[SWaTConfig],
    Field(discriminator='type')
]

# =========
# Generator
# =========
class LSTMGeneratorConfig(BaseModel):
    type: Literal['lstm']
    in_dim: int
    out_dim: int
    hidden_size: int
    num_layers: int

GeneratorConfig = Annotated[
    Union[LSTMGeneratorConfig],
    Field(discriminator='type')
]

# =============
# Discriminator
# =============
class LSTMDiscriminatorConfig(BaseModel):
    type: Literal['lstm']
    in_dim: int
    hidden_size: int
    num_layers: int

DiscriminatorConfig = Annotated[
    Union[LSTMDiscriminatorConfig],
    Field(discriminator='type')
]

# =====
# Model
# =====
class ReverseMapConfig(BaseModel):
    type: Literal['reverse_map']
    # models
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig
    # losses
    loss: str = 'bce'
    gp_weight: float = 10.
    # stabilization techiques
    disc_real_label: float = 1.0 # used for label smoothing in BCE
    clip_grad_g: float = 0.0
    clip_grad_d: float = 0.0
    bounded_dequantization: float = 0.0 # used for actuators if present
    # optimizer setup
    lr_g: float
    lr_d: float
    betas_g: list[float] = [0.5, 0.999]
    betas_d: list[float] = [0.5, 0.999]
    # training ratio
    generator_rounds: int
    discriminator_rounds: int

ModelConfig = Annotated[
    Union[ReverseMapConfig],
    Field(discriminator='type')
]

# ======
# Params
# ======
class Params(BaseModel):
    window_size: int
    stride: int
    batch_size: int
    num_workers: int
    shuffle: bool

class Settings(BaseModel):
    paths: Paths
    dataset: DatasetConfig
    model: ModelConfig
    params: Params
    n_epochs: int
    device_name: str


def load_settings(settings_file: Path, experiment_name: str, n_epochs: int) -> Settings:
    with open(settings_file, 'rb') as file:
        config = tomllib.load(file)

    config['paths'] = {
        'data_root': Path(constants.DATA_ROOT),
        'state_dir': Path(constants.STATES_ROOT) / experiment_name,
    }
    config['n_epochs'] = n_epochs
    config['device_name'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    return Settings.model_validate(config)
