import numpy as np
import torch
import torch.utils.data
import torchvision
import glob
from einops import rearrange

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
