import os
import random
from typing import List
import argparse

import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import OmegaConf
import wandb

from dataset.dataset import Dataset
from train import train_model
from models.model_v1 import Model

# This is a simple dictionary that maps, for each of the domains D1,D2,D3, to their corresponding data folder(s)
DATA_PATH = '/data-local/data1-ssd/dpujolpe/EpicKitchens/EPIC-KITCHENS'
LABEL_PATH = '/data-local/data1-ssd/dpujolpe/EpicKitchens/EPIC-KITCHENS/annotations'
FILE_NAMES = {'train': 'EPIC_100_train.pkl',
              'val': 'EPIC_100_validation.pkl',
              'test': 'EPIC_100_test_timestamps.pkl'}

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_experiment(cfg: OmegaConf) -> None:

    # Set random seed
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed_all(cfg.train.seed)
    torch.set_default_tensor_type(torch.FloatTensor)

     # Define the device
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available:
        DEVICE = torch.device('cuda')

        # Set the device
        torch.cuda.set_device(5) # opt.runtime.gpu

    model = Model(cfg.model)
    model = model.to(DEVICE)

    print('Loading data...')
    batch_size = cfg.train.BATCH_SIZE
    data_threads = cfg.train.DATA_THREADS  # These are the number of workers to use for the data loader

    print("Loading the training dataset")

    train_path = os.path.join(DATA_PATH, 'train')
    train_set = Dataset(data_dirs = [train_path])

    train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, batch_size=batch_size,
                                                num_workers=data_threads, drop_last=True, pin_memory=True)

    sample, label = train_loader.__get_item__()

    print(f'Sample type: {type(sample)}')
    print(f'Sample shape: {sample.shape}')
    print(f'Sample label: {label}')