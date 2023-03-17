import os
import random
from typing import List
import argparse

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra import compose, initialize

from tasks.EpicKitchens.dataset.dataset import Dataset
from Domain_adaptive_WTAL.src.utils.misc import collate_fn

from models import model_v1

# This is a simple dictionary that maps, for each of the domains D1,D2,D3, to their corresponding data folder(s)
DATA_PATH = '/data-local/data1-ssd/dpujolpe/EpicKitchens/EPIC-KITCHENS'
LABEL_PATH = '/data-local/data1-ssd/dpujolpe/EpicKitchens/EPIC-KITCHENS/annotations'
FILE_NAMES = {'train': 'EPIC_100_train.pkl',
              'val': 'EPIC_100_validation.pkl',
              'test': 'EPIC_100_test_timestamps.pkl'}

# This shouldn't be here, but we leave it here for the moment until we move it
@hydra.main(version=None, config_path='./configs', config_name='model_v1')
def cfg_setup(cfg: DictConfig):
    return cfg

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def create_options():
    parser = argparse.ArgumentParser()

    # ========================= Runtime Configs ==========================
    parser.add_argument('--seed', default=0, type=int,
                        help='manual seed')
    parser.add_argument('--data_threads', type=int, default=5,
                        help='number of data loading threads')

    # ========================= Model Configs ==========================
    parser.add_argument('--dropout_rate', default=0.5, type=float,
                        help='dropout ratio for frame-level feature (default: 0.5)')

    # ========================= Learning Configs ==========================
    parser.add_argument('--epochs', default=1000, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='-batch size')

    opt = parser.parse_args()
    return opt


def run_experiment(name: str, pretrained_state_path:str = None) -> None:
    opt = create_options()

    wandb.init(project=name)
    working_directory = wandb.run.dir

    # Set all the random seeds
    random.seed(opt.runtime.seed)
    torch.manual_seed(opt.runtime.seed)
    torch.cuda.manual_seed_all(opt.runtime.seed)
    torch.set_default_tensor_type(torch.FloatTensor)


    # Subdirectory for the pretrained model (if needed)
    pretrain_exp = os.path.join(working_directory, 'saved_models')
    if not os.path.isdir(pretrain_exp):
        os.makedirs(pretrain_exp)

    # Define the device
    DEVICE = torch.device('cpu')
    if opt.runtime.use_gpu:
        DEVICE = torch.device('cuda')

        # Set the device
        torch.cuda.set_device(5)#opt.runtime.gpu)


    # Create the model with the given options
    samp_rate=0.5   # This implies sampling every other frame

    # TODO: Add here the initialization of your model
    # This is our general model, even though we may have different configurations (depending on what

    # Send it to the desired device
    model = model.to(DEVICE)

    # Load a pretrained model, if specified
    start_epoch = 1
    if pretrained_state_path is not None and os.path.isfile(pretrained_state_path):
        print('Loading the model and optimizer')
        model, optimizer, start_epoch = load_ckp(pretrained_state_path, model, optimizer)


    # -------------------------------------------------
    print("Loading the data...")

    batch_size = opt.batch_size
    data_threads = opt.data_threads    # These are the number of workers to use for the data loader

    # Load the source training domain
    print("Loading the training dataset")

    train_path = os.path.join(DATA_PATH, 'train')
    train_set = Dataset(data_dirs = [train_path])

    train_sampler = torch.utils.data.sampler.RandomSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler, batch_size=batch_size,
                                                num_workers=data_threads, collate_fn=collate_fn, drop_last=True, pin_memory=True)

    # Load the validation clips (this is the data that we test it with
    print("Loading the validation dataset")
    val_path = os.path.join(DATA_PATH, 'val')
    val_set =  Dataset(data_dirs = [val_path])

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=data_threads, collate_fn=collate_fn, drop_last=True,
                                             pin_memory=True)

    wandb.config.update(opt)

    # TODO: Run here the training function of your model


    # Stop the logging
    wandb.finish()



if __name__ == '__main__':
    # Create the experiment
    run_experiment(name='action_classificatoin')
