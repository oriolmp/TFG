import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import os
from einops import rearrange
import pandas as pd
from PIL import Image

# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_dirs):
#         super().__init__()

#         self.data_dirs = data_dirs
#         self.clip_paths =sum([glob.glob(dir + '/*.pt') for dir in data_dirs], [])

#     def __len__(self):
#         return len(self.clip_paths)

#     def __getitem__(self, index):
#         file_path = self.clip_paths[index]
#         data = torch.load(file_path,map_location=torch.device('cpu'))
#         # data = torch.load(self.data_dirs[index],map_location=torch.device('cpu'))
    
#         # Process the data if necessary
#         video_data = data['data'].squeeze(0)    # shape: T x C

#         labels = data['labels']
        
#         return video_data, labels

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
        self.clips_df = self.clips_df.drop(self.clips_df[self.clips_df['video_id'] == 'P23_04'].index)
    
    def __len__(self):
        return self.clips_df.shape[0]

    def __getitem__(self, index):
        clip_info = self.clips_df.iloc[index]
        clip_dir = self.frames_dir + clip_info['participant_id'] + '/rgb_frames/' + clip_info['video_id']

        resize = T.Resize(size=(self.frame_size, self.frame_size), antialias=True) # antialias=True because of user warning
        
        # oad all frames and do preprocessing
        frame_paths = [clip_dir + '/frame_' + str(x).rjust(10, '0') + '.jpg' for x in range(clip_info['start_frame'], clip_info['stop_frame'])]
        total_frames = len(frame_paths)   

        # Load rgb frames with shape (1920, 1080, 3). 
        frames = [torch.tensor(np.asarray(Image.open(x))).to('cpu') for x in frame_paths if os.path.isfile(x)]
        
        if len(frames) != clip_info['stop_frame'] - clip_info['start_frame']:
            exit
        
        # rearrange to allow resize
        frames = [rearrange(x, 'h w c -> c h w') for x in frames]
        # Resize it to FRAME_SIZE and add temporal dimension: (3, 224, 224, 1)
        frames = [resize(x).unsqueeze(-1) for x in frames]
         # Concat all frames. Shape: (3, 224, 224, frames)
        clip = torch.cat(frames, dim=-1)

        # apply padding if total frames aren't enough
        if total_frames < self.num_frames:
            missing_frames = self.num_frames - total_frames

            # check if missing frames is odd in order to ensure that after padding, 
            # num_frames equals NUM_FRAMES
            if missing_frames % 2 == 0:
                pad = (missing_frames // 2, missing_frames // 2)
            else:
                pad = (missing_frames // 2 + 1, missing_frames // 2)
            
            clip = F.pad(clip, pad, 'constant', 0)
        
        # apply random sampling or uniform sampling
        # uniform
        elif total_frames > self.num_frames:
            range_frame = total_frames - 200
            rand_frame = torch.randint(low=1, high=range_frame, size=(1,))
            clip = rearrange(clip, 'c w h t1 -> t1 c w h')
            clip = clip[rand_frame:rand_frame+200]
            clip = rearrange(clip, 't2 c w h -> c w h t2')   

        # rearrange to fit model
        clip = rearrange(clip, 'c w h t -> c t w h')

        label = clip_info['verb_class']
        
        return clip, label