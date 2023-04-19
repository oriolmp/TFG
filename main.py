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
DATA_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/'
LABEL_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/labels'
ANNOTATIONS_NAMES = {'train': 'EPIC_100_train.csv',
                     'val': 'EPIC_100_validation.csv',
                     'test': 'EPIC_100_test_timestamps.csv'}

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
    working_directory = wandb.run.dir

    # Set all the random seeds
    random.seed(cfg.training.SEED)
    torch.manual_seed(cfg.training.SEED)
    torch.cuda.manual_seed_all(cfg.training.SEED)
    torch.set_default_tensor_type(torch.FloatTensor)

    # Subdirectory for the pretrained model (if needed)
    pretrain_exp = os.path.join(working_directory, 'weights')
    if not os.path.isdir(pretrain_exp):
        os.makedirs(pretrain_exp)

    # Define the device
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available:
        DEVICE = torch.device('cuda')
        # Set the device
        torch.cuda.set_device(5) 

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
    
    # wandb.config.update(opt)
    
    # Load the source training domain
    print("Loading the training dataset")

    train_path = os.path.join(DATA_PATH, 'train/')
    annotations_path = os.path.join(LABEL_PATH, ANNOTATIONS_NAMES['train'])
    train_set = Dataset(cfg, frames_dir=train_path, annotations_file=annotations_path)

    train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, batch_size=batch_size,
                                                num_workers=data_threads, drop_last=True, pin_memory=True)

    # Load the validation clips (this is the data that we test it with)
    print("Loading the validation dataset")
    val_path = os.path.join(DATA_PATH, 'val/')
    annotations_path = os.path.join(LABEL_PATH, ANNOTATIONS_NAMES['val'])
    val_set =  Dataset(cfg, frames_dir=val_path, annotations_file=annotations_path)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=data_threads, drop_last=True,
                                             pin_memory=True)
    
    dataloaders = [train_loader, val_loader]

    params_to_update = model.parameters()
    optimizer = optim.Adam(params_to_update)
    criterion = nn.CrossEntropyLoss()

    num_epochs = cfg.training.EPOCHS
    print_batch = cfg.training.PRINT_BATCH
    trained_model = train_model(model, dataloaders, criterion, optimizer, DEVICE, num_epochs, print_batch)

    # Save model
    i = 1
    save_model_path = pretrain_exp + f'model_{i}'
    while os.path.isdir(save_model_path):
        i += 1
        save_model_path = pretrain_exp + f'model_{i}'
    torch.save(trained_model.state_dict(), save_model_path)

    # Stop the logging
    wandb.finish()


if __name__ == '__main__':

    # Create the experiment
    run_experiment()
