from dataset.dataset import Dataset
from datetime import datetime
from omegaconf import OmegaConf
from einops import rearrange
import numpy as np
import os
import hydra
import torch
import cv2
import time
import pandas as pd
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from random import randint

DATA_PATH = '/data-fast/107-data4/omartinez/FULL_EPIC_KITCHENS'
LABEL_PATH = '/data-fast/107-data4/omartinez/FULL_EPIC_KITCHENS/labels'
CUSTOM_LABEL_PATH = '/home-net/omartinez/TFG/custom_sets/'
ANNOTATIONS_NAMES = {'train': 'subset_train.csv',
                     'val': 'subset_val.csv',
                     'test': 'EPIC_100_validation.csv'}
SAMPLE_DIRECTORY = '/home-net/omartinez/TFG/samples/'

os.environ["HYDRA_FULL_ERROR"] = "1"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, frames_dir, annotations_file):
        """
        Gets frames from each video, generating its torch tensor.
        The classes are extracted from the csv containing all labels and start and end frames.
        We will use cfg to set number of frames and its resolution.

        Args:
            frame_dir: path containing all frames. Its structure is /train/PXX/rgb_frames/PXX_YY
            info_dir: path to csv file containing video info (class, start_frame, end_frame...)
            cfg: used to refer to dataset.yaml to configure the preprocess
        """
        super().__init__()

        self.num_frames = cfg.dataset.NUM_FRAMES
        self.frame_size = cfg.dataset.FRAME_SIZE
        self.frames_dir = frames_dir
        info_df = pd.read_csv(annotations_file)
        self.clips_df = info_df[['participant_id', 'video_id', 'start_frame', 'stop_frame', 'verb_class']]
        # drop video with missing frames
        # if using cutsom sets, this video is directly removed
        # self.clips_df = self.clips_df.drop(self.clips_df[self.clips_df['video_id'] == 'P23_04'].index)
    
    def __len__(self):
        return self.clips_df.shape[0]

    def __getitem__(self, index):
        print('VIDEO')
        start_time = time.time()
        clip_info = self.clips_df.iloc[index]
        clip_dir = self.frames_dir + clip_info['participant_id'] + '/rgb_frames/' + clip_info['video_id']
        print(f'\tExtract df info: {time.time() - start_time}')

        start_time = time.time()
        resize = T.Resize(size=(self.frame_size, self.frame_size), antialias=True) # antialias=True because of user warning
        to_tensor = T.ToTensor()
        print(f'\tInitialize transforms functions: {time.time() - start_time}')
        
        # Load all frames and do preprocessing
        start_time = time.time()
        frame_paths = [clip_dir + '/frame_' + str(x).rjust(10, '0') + '.jpg' for x in range(clip_info['start_frame'], clip_info['stop_frame'])]
        total_frames = len(frame_paths)   
        print(f'\tGenerate frame paths list: {time.time() - start_time}')

        # Load rgb frames with shape (3, 1920, 1080). 
        print(f'\tTotal frames: {total_frames}')
        start_time = time.time()
        # frames = [to_tensor(Image.open(x)) for x in frame_paths if os.path.isfile(x)]
        frames = [to_tensor(cv2.imread(x, cv2.IMREAD_UNCHANGED)) for x in frame_paths if os.path.isfile(x)]
        load_time = time.time() - start_time
        print(f'\tRead all frames: {load_time}')
        print(f'\tTime load each frame: {load_time/total_frames}')

        if len(frames) != clip_info['stop_frame'] - clip_info['start_frame']:
            exit
        
        # Resize it to FRAME_SIZE and add temporal dimension: (3, 112, 112, 1)
        start_time = time.time()
        frames = [resize(x).unsqueeze(-1) for x in frames]
        print(f'\tResize and unsqueeze: {time.time() - start_time}')
         # Concat all frames. Shape: (3, 112, 112, frames)
        start_time = time.time()
        clip = torch.cat(frames, dim=-1)
        print(f'\tConcat frames: {time.time() - start_time}')

        # apply padding if total frames aren't enough
        if total_frames <= 2 * self.num_frames:
            start_time = time.time()
            missing_frames = 2 * self.num_frames - total_frames

            # check if missing frames is odd in order to ensure that after padding, 
            # num_frames equals NUM_FRAMES
            if missing_frames % 2 == 0:
                pad = (missing_frames // 2, missing_frames // 2)
            else:
                pad = (missing_frames // 2 + 1, missing_frames // 2)
            
            clip = F.pad(clip, pad, 'constant', 0)
            clip = rearrange(clip, 'c w h t1 -> t1 c w h')
            clip = clip[0:2*self.num_frames:2] # take 1 for every 2 frames
            clip = rearrange(clip, 't2 c w h -> c w h t2')
            print(f'\tPad clip: {time.time() - start_time}')
     
        
        # apply  uniform sampling      
        elif total_frames > 2 * self.num_frames:
            start_time = time.time()
            range_frame = total_frames - 2 * self.num_frames
            rand_frame = torch.randint(low=0, high=range_frame, size=(1,)) + 1 # we add + 1 since frame_0000000 does not exists
            clip = rearrange(clip, 'c w h t1 -> t1 c w h')
            clip = clip[rand_frame:rand_frame+2*self.num_frames:2] # take 1 for every 2 frames
            clip = rearrange(clip, 't2 c w h -> c w h t2')
            print(f'\tSelect frames: {time.time() - start_time}')   

        # rearrange to fit model
        start_time = time.time()
        clip = rearrange(clip, 'c w h t -> c t w h')
        print(f'\tRearrange clip: {time.time() - start_time}')

        label = clip_info['verb_class']
        
        return clip, label

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_debug(cfg: OmegaConf):

    train_path = os.path.join(DATA_PATH, 'train/')
    annotations_path = os.path.join(CUSTOM_LABEL_PATH, ANNOTATIONS_NAMES['train'])

    batch_size = cfg.training.BATCH_SIZE
    data_threads = cfg.training.DATA_THREADS

    start_time = time.time()

    train_set = Dataset(cfg, frames_dir=train_path, annotations_file=annotations_path)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                                                num_workers=data_threads, drop_last=True, pin_memory=True)
    
    print(f'Elapsed time to initialize data classes: {time.time() - start_time}')

    # for i in range(0,10):
    #     start_time = time.time()
    #     idx = randint(a=1, b=100)
    #     _ = train_set.__getitem__(idx)
    #     elapsed_time = time.time() - start_time
    #     print(f'\tGet item total time: {elapsed_time}')


    before_for = time.time()
    for clips, labels in train_loader:  
        print(f'Time to load batch clips: {time.time() - before_for}')
        break


if __name__ == '__main__': 
    print('Debug starting...')   
    run_debug()
