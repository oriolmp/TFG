import os
import random
import time
from typing import List

import hydra
import torch
import wandb
from hydra import compose, initialize
from einops import rearrange

from utils.misc import collate_fn


# This is a simple dictionary that maps, for each of the domains D1,D2,D3, to their corresponding data folder(s)
DATA_PATH = '/data-net/datasets/EpicKitchens/EPIC-KITCHENS'
LABEL_PATH = '/data-net/datasets/EpicKitchens/EPIC-KITCHENS/labels'
FILE_NAMES = {'train': 'EPIC_100_train.pkl',
              'val': 'EPIC_100_validation.pkl',
              'test': 'EPIC_100_test_timestamps.pkl'}

def create_transformation():
    from tasks.EpicKitchens.data_downloading.e2e_lib.videotransforms_gpu import GroupRandomHorizontalFlip, GroupRandomCrop, GroupCenterCrop, \
            GroupPhotoMetricDistortion, GroupRotate, GroupResize, GroupResizeShorterSide, GroupNormalize
    import torchvision.transforms as transforms

    data_transform = transforms.Compose([
        GroupResize((224, 224)),
        GroupNormalize(to_rgb=True)
    ])
    return data_transform


def apply_backbone(x, backbone):
    # Shape: BxCxT
    i3d_type = 'frame_level'
    b, c, t, h, w = x.shape
    if i3d_type == 'frame_level':
        padding = torch.zeros(b, c, 8, h, w).to(x.device)
        x = torch.cat((padding, x, padding), 2).to(x.device)  # b c t' h w
        result = torch.Tensor().to(x.device)
        for i in range(t):
            frame_batch = x[:, :, i:i + 16, :, :]  # b c 16 h w  -> b c
            frame_feats = backbone.extract_features(frame_batch).unsqueeze(-1)
            result = torch.cat((result, frame_feats), 2)
        return result

    if i3d_type == 'static':
        result = torch.Tensor().to(x.device)
        for i in range(t):
            frame_batch = x[:, :, i, :, :].unsqueeze(2).repeat(1, 1, 16, 1, 1)  # b c 16 h w
            frame_feats = backbone.extract_features(frame_batch).unsqueeze(-1)  # b c 1
            result = torch.cat((result, frame_feats), 2)
        return result
    if i3d_type == 'static_old':
        b, c, t, h, w = x.shape
        x_frame = rearrange(x, 'b c t h w -> (b t) c h w')
        x_frame = backbone.extract_features(x_frame.unsqueeze(2).repeat(1, 1, 16, 1, 1))
        return rearrange(x_frame, '(b t) c -> b c t', b=b)


def preprocess_dataset(data_path=None, output_dir:str = None):
    # Set the cuda device
    torch.cuda.set_device(3)

    # -------------------------------------------------
    print("Loading the data...")

    data_threads = 8

    # Load the source training domain
    print("Loading the training dataset")
    train_src_path = os.path.join(DATA_PATH, 'train', src_domain, 'rgb_frames')

    info_path = os.path.join(
        '/home-net/dpujolpe/Domain_adaptive_WTAL/Domain_adaptive_WTAL/src/tasks/Epic_kitchens_100vs100/video_info',
        'video_info_train')
    label_path = os.path.join(LABEL_PATH, FILE_NAMES['train'])
    train_source_set = build_dataset(data_path=train_src_path,
                                    info_path=info_path,
                                    label_path=label_path,
                                    get_labels=True,
                                    mode='train',
                                    samp_rate=0.5)
    train_src_loader = torch.utils.data.DataLoader(train_source_set, batch_size=1, num_workers=data_threads, collate_fn=collate_fn, drop_last=True, pin_memory=True)


    # Create the data transformation
    transformation = create_transformation()


    # Load the I3D model
    from Domain_adaptive_WTAL.src.models.backbone.models.pytorch_i3d import InceptionI3d
    backbone = InceptionI3d(400, in_channels=3).cuda()
    backbone.load_state_dict(torch.load('/home-net/dpujolpe/Domain_adaptive_WTAL/Domain_adaptive_WTAL/src/models/backbone/pretrained_models/I3D/rgb_imagenet.pt'))
    backbone.eval()
    backbone.num_channels = 1024
    backbone.requires_grad_(False)

    # Create the output directory
    output_dir = os.path.join(output_dir, src_domain)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # TODO: Iterate over each of the segments of the label set. Then apply I3D over their corresponding RGB. Then save the extracted features
    # Now process the train dataset
    output_dir_train = os.path.join(output_dir, 'train')
    if not os.path.exists(output_dir_train):
        os.makedirs(output_dir_train)

    for i, (batch, label) in enumerate(train_src_loader):
        # Load the target input (shape BxCxTxHxW) and the annotations
        input = batch.tensors.cuda()  # Load the target data
        label = label[0]

        # Apply the data augmentations
        input = apply_transformation(input, transforms=transformation)

        # Apply a pretrained I3D model (on Imagenet).
        # Apply a backbone so that the input data has shape BxCxTxHxW -> B x F' x T
        input = apply_backbone(input, backbone)

        input = rearrange(input, 'b f t -> b t f')

        data = {'data': input,
                'label': label}

        # Save the preprocessed features into a .pt file
        cut_path = os.path.join(output_dir_train, 'cut_' + str(i)+'.pt')
        torch.save(data, cut_path)


# Preprocess domain P01
preprocess_dataset(data_path='', output_dir='/data-local/data1-ssd/dpujolpe/Processed_EK100_I3D_frame')

