import os
import random
from typing import List

import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import OmegaConf

from dataset.dataset import Dataset

# We set this variable since it raises an error if not
os.environ["HYDRA_FULL_ERROR"] = "1"

# This is a simple dictionary that maps, for each of the domains D1,D2,D3, to their corresponding data folder(s)
DATA_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/'
LABEL_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/labels'

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_experiment(cfg: OmegaConf) -> None:

    print(f'cfg: {cfg}')

    # Set random seed
    random.seed(cfg.training.SEED)
    torch.manual_seed(cfg.training.SEED)
    torch.cuda.manual_seed_all(cfg.training.SEED)
    torch.set_default_tensor_type(torch.FloatTensor)

     # Define the device
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available:
        print('Cuda available!')
        DEVICE = torch.device('cuda')

        # Set the device
        torch.cuda.set_device(5) 

    print(f'Device: {DEVICE}')

    print("Loading the training dataset")

    train_path = os.path.join(DATA_PATH, 'train/')
    annotations_path = os.path.join(LABEL_PATH, 'EPIC_100_train.csv')
    train_set = Dataset(cfg, frames_dir=train_path, annotations_file=annotations_path)

    idx = 0
    sample, label = train_set.__getitem__(idx)

    print(f'Sample type: {type(sample)}')
    print(f'Sample shape: {sample.shape}')
    print(f'Sample label: {label}')

run_experiment()