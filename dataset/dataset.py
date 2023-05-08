import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import os
from einops import rearrange
import pandas as pd
# from PIL import Image
import cv2


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
        clip_info = self.clips_df.iloc[index]
        clip_dir = self.frames_dir + clip_info['participant_id'] + '/rgb_frames/' + clip_info['video_id']

        resize = T.Resize(size=(self.frame_size, self.frame_size), antialias=True) # antialias=True because of user warning
        to_tensor = T.ToTensor()
        
        # Load all frames and do preprocessing
        frame_paths = [clip_dir + '/frame_' + str(x).rjust(10, '0') + '.jpg' for x in range(clip_info['start_frame'], clip_info['stop_frame'])]
        total_frames = len(frame_paths)   

        # Load rgb frames with shape (3, 1920, 1080). 
        # frames = [PIL_to_tensor(Image.open(x)) for x in frame_paths if os.path.isfile(x)]
        frames = [to_tensor(cv2.imread(x, cv2.IMREAD_UNCHANGED)) for x in frame_paths if os.path.isfile(x)]

        if len(frames) != clip_info['stop_frame'] - clip_info['start_frame']:
            exit
        
        # Resize it to FRAME_SIZE and add temporal dimension: (3, 112, 112, 1)
        frames = [resize(x).unsqueeze(-1) for x in frames]
         # Concat all frames. Shape: (3, 112, 112, frames)
        clip = torch.cat(frames, dim=-1)

        # apply padding if total frames aren't enough
        if total_frames <= 2 * self.num_frames:
            missing_frames = self.num_frames - total_frames

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
        
        # apply  uniform sampling      
        elif total_frames > 2 * self.num_frames:
            range_frame = total_frames - 2 * self.num_frames
            rand_frame = torch.randint(low=0, high=range_frame, size=(1,)) + 1 # we add + 1 since frame_0000000 does not exists
            clip = rearrange(clip, 'c w h t1 -> t1 c w h')
            clip = clip[rand_frame:rand_frame+2*self.num_frames:2] # take 1 for every 2 frames
            clip = rearrange(clip, 't2 c w h -> c w h t2')   

        # rearrange to fit model
        clip = rearrange(clip, 'c w h t -> c t w h')

        label = clip_info['verb_class']
        
        return clip, label