from dataset.dataset import Dataset
from datetime import datetime
from omegaconf import OmegaConf
from einops import rearrange
import numpy as np
import os
import hydra
import torch
import cv2

DATA_PATH = '/data-fast/107-data4/omartinez/FULL_EPIC_KITCHENS'
LABEL_PATH = '/data-fast/107-data4/omartinez/FULL_EPIC_KITCHENS/labels'
CUSTOM_LABEL_PATH = '/home-net/omartinez/TFG/custom_sets/'
ANNOTATIONS_NAMES = {'train': 'subset_train.csv',
                     'val': 'subset_val.csv',
                     'test': 'EPIC_100_validation.csv'}
SAMPLE_DIRECTORY = '/home-net/omartinez/TFG/samples/'

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_debug(cfg: OmegaConf):

    train_path = os.path.join(DATA_PATH, 'train/')
    annotations_path = os.path.join(CUSTOM_LABEL_PATH, ANNOTATIONS_NAMES['train'])

    batch_size = cfg.training.BATCH_SIZE
    data_threads = cfg.training.DATA_THREADS

    train_set = Dataset(cfg, frames_dir=train_path, annotations_file=annotations_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                                num_workers=data_threads, drop_last=True, pin_memory=True)

    for clips, labels in train_loader:  
        for i in range(cfg.training.BATCH_SIZE):
            sample_clip = clips[i]
            sample_label = labels[i]

            print(f'sample label {i}: {sample_label}')
            
            frames_tensor = rearrange(sample_clip, 'c t w h -> t w h c')
            frames_array = frames_tensor.numpy()

            video_name = f'sample_{i}.avi'
            video_path = os.path.join(SAMPLE_DIRECTORY, video_name)
            print(f'Generating video at {video_path}')
            writer = cv2.VideoWriter(
                filename=video_path, 
                fourcc=cv2.VideoWriter_fourcc(*'DIVX'), 
                fps=15, 
                frameSize=(112, 112)
            )
            for i in range(len(frames_array)):
                # convert to right format
                img = np.uint8(frames_array[i] * 255)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img)
            
            print('video done!')
            
            writer.release()
        break


if __name__ == '__main__': 
    print('Debug starting...')   
    run_debug()
