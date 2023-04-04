import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import glob
from einops import rearrange
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs):
        super().__init__()

        self.data_dirs = data_dirs
        self.clip_paths =sum([glob.glob(dir + '/*.pt') for dir in data_dirs], [])

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, index):
        file_path = self.clip_paths[index]
        data = torch.load(file_path,map_location=torch.device('cpu'))
        # data = torch.load(self.data_dirs[index],map_location=torch.device('cpu'))
    
        # Process the data if necessary
        video_data = data['data'].squeeze(0)    # shape: T x C

        labels = data['labels']
        
        return video_data, labels

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
        self.clips_df = info_df[['partcipant_id', 'video_id', 'start_frame', 'stop_frame', 'verb_class']]

    def __len__(self):
        return len(self.clip_df.shape[0])

    def __getitem__(self, index):
        clip_info = self.clip_df.iloc[index]
        clip_dir = clip_info['participant_id'] + '/rgb_frames/' + clip_info['video_id']

        resize = T.Resize(size=(self.frame_size, self.frame_size))

        # First idea: only load some frames to lighten computation
        # middel_frame = clip_info['start_frame'] + (clip_info['stop_frame'] - clip_info['start_frame']) // 2
        # clip_frames = (middel_frame - self.num_frames // 2, middel_frame + self.num_frames // 2)
        
        # Second idea: load all frames and do preprocessing
        frame_paths = [clip_dir + 'frame_' + str(x).rjust(10, '0') + '.jpg' for x in range(clip_info['start_frame'] , clip_info['stop_frame'])]
        total_frames = len(frame_paths)   

        # Load rgb image with shape (3, 1920, 1080). 
        # Resize it to FRAME_SIZE: (3, 224, 224)
        # Unsqueeze it to add temporal dimension. Shape: (3, 1, 224, 224)
        # Concat all frames. Shape: (3, frames, 1920, 1080)
        frames = [resize(torch.load(x, map_location=torch.device('cpu'))).unsqueeze(1) for x in frame_paths]
        video = torch.cat(frames)

        if total_frames < self.num_frames:
            missing_frames = self.num_frames - total_frames

            # check if missing frames is odd in order to ensure that after padding, 
            # num_frames equals NUM_FRAMES
            if missing_frames % 2 == 0:
                pad = (0, 0, 0, 0, missing_frames // 2, missing_frames // 2)
            else:
                pad = (0, 0, 0, 0, missing_frames // 2 + 1, missing_frames // 2)
            
            video = F.pad(video, pad, 'constant', 0)
        elif total_frames > self.num_frames:
            
    
        # Process the data if necessary
        video_data = data['data'].squeeze(0)    # shape: T x C

        labels = data['labels']
        
        return video_data, labels