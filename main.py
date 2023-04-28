import os
import random
from typing import List
from datetime import datetime

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from omegaconf import OmegaConf
import wandb

from dataset.dataset import Dataset
from train import train_model
from models.model_v1 import Model


# This is a simple dictionary that maps, for each of the domains D1,D2,D3, to their corresponding data folder(s)
DATA_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS/'
LABEL_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS/labels'
CUSTOM_LABEL_PATH = '/home-net/omartinez/custom_sets/'
ANNOTATIONS_NAMES = {'train': 'train.csv',
                     'val': 'val.csv',
                     'test': 'EPIC_100_validation.csv'}
WEIGHTS_DIR = '/home-net/omartinez/TFG/weights/'

# We set this variable since it raises an error if not
os.environ["HYDRA_FULL_ERROR"] = "1"

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']



@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_experiment(cfg: OmegaConf) -> None:

    # print cfg file
    print(OmegaConf.to_yaml(cfg))

    wandb.init(project=cfg.NAME)
    # working_directory = wandb.run.dir

    # Set all the random seeds
    random.seed(cfg.training.SEED)
    torch.manual_seed(cfg.training.SEED)
    torch.cuda.manual_seed_all(cfg.training.SEED)
    torch.set_default_tensor_type(torch.FloatTensor)

    # Subdirectory for the pretrained model (if needed)
    if not os.path.isdir(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)

    # Define the device
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available:
        DEVICE = torch.device('cuda')
        torch.cuda.set_device(cfg.training.GPU) 

    # This is our general model, even though we may have different configurations (depending on what
    model = Model(cfg)
    
    # Send it to the desired device
    model = model.to(DEVICE)

    # Load a pretrained model, if specified
    start_epoch = 1
    if cfg.training.PRETRAINED_STATE_PATH is not None and os.path.isfile(cfg.training.PRETRAINED_STATE_PATH):
        print('Loading the model and optimizer')
        model, optimizer, start_epoch = load_ckp(cfg.training.PRETRAINED_STATE_PATH, model, optimizer)


    # -------------------------------------------------
    print("Loading the data...")

    batch_size = cfg.training.BATCH_SIZE
    data_threads = cfg.training.DATA_THREADS  # These are the number of workers to use for the data loader
    
    # Load the source training domain
    print("Loading the training dataset")

    train_path = os.path.join(DATA_PATH, 'train/')
    annotations_path = os.path.join(CUSTOM_LABEL_PATH, ANNOTATIONS_NAMES['train'])
    train_set = Dataset(cfg, frames_dir=train_path, annotations_file=annotations_path)

    train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, batch_size=batch_size, shuffle=True, 
                                                num_workers=data_threads, drop_last=True, pin_memory=True)

    # Load the validation clips (this is the data that we test it with)
    print("Loading the validation dataset")
    val_path = os.path.join(DATA_PATH, 'val/')
    annotations_path = os.path.join(CUSTOM_LABEL_PATH, ANNOTATIONS_NAMES['val'])
    val_set =  Dataset(cfg, frames_dir=val_path, annotations_file=annotations_path)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=data_threads, drop_last=True,
                                             pin_memory=True)
    
    dataloaders = [train_loader, val_loader]

    num_epochs = cfg.training.EPOCHS
    print_batch = cfg.training.PRINT_BATCH
    lr = cfg.training.LEARNING_RATE 

    params_to_update = model.parameters()
    optimizer = optim.Adam(params=params_to_update, lr=lr)

    criterion = nn.CrossEntropyLoss()

    trained_model = train_model(model, dataloaders, criterion, optimizer, DEVICE, num_epochs, print_batch)

    # Save model
    i = 1
    save_model_path = WEIGHTS_DIR + cfg.model.ATTENTION + f'_{i}'
    while os.path.isdir(save_model_path):
        i += 1
        save_model_path = WEIGHTS_DIR + cfg.model.ATTENTION + f'_{i}'
    torch.save(trained_model.state_dict(), save_model_path)
    print(f'Model saved at {save_model_path}')

    # Stop the logging
    wandb.finish()


if __name__ == '__main__':

    time = datetime.now()
    print('Training starting...')
    print(f'Datetime: {time}')
    
    # Create the experiment
    run_experiment()
