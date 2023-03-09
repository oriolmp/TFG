import os
import random
import time
from typing import List

import csv
import hydra
import torch
import wandb
from hydra import compose, initialize
from einops import rearrange
from utils.misc import collate_fn

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


def load_video_frames(frame_dir,
                      start,
                      end,
                      fn_tmpl='img_%07d.jpg',
                      samp_rate:float=0.5):
    '''
    Load a sequence of video frames into memory, while discarding (1-samp_rate) frames

    Params:
        frame_dir: the directory to the decoded video frames
        start: load image starting with this index. 1-indexed
        end: length of the loaded sub-sequence.
        stride: load one frame every `stride` frame from the sequence.
    Returns:
        Nd-array with shape (T, H, W, C) in float32 precision. T = num // stride
    '''
    # Read the images in parallel

    # Calculate the stride, depending on the sampling rate
    stride = int(1 / samp_rate)

    img_paths = [os.path.join(frame_dir, fn_tmpl % i) for i in range(start, end, stride)]
    frames = [cv2.imread(path) for path in img_paths]

    return np.asarray(frames, dtype=np.float32)  # T x H x W x C


def preprocess_dataset(data_path=None, label_path=None, output_dir:str = None):
    # Set the cuda device
    torch.cuda.set_device(3)

    # Create the output directory
    output_dir = os.path.join(output_dir, src_domain)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create the data transformation
    transform = create_transformation()

    # Load the I3D model to extract features
    from Domain_adaptive_WTAL.src.models.backbone.models.pytorch_i3d import InceptionI3d
    backbone = InceptionI3d(400, in_channels=3).cuda()
    backbone.load_state_dict(torch.load('/home-net/dpujolpe/Domain_adaptive_WTAL/Domain_adaptive_WTAL/src/models/backbone/pretrained_models/I3D/rgb_imagenet.pt'))
    backbone.eval()
    backbone.num_channels = 1024
    backbone.requires_grad_(False)

    # Iterate over each of the segments
    with open(label_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Iterate over each of the segments
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']

            verb_class = row['verb_class']
            noun_class = row['noun_class']

            participant_id = row['participant_id']
            video_id = row['video_id']
            narration_id = row['narration_id']

            # Now read the RGBs of this segments
            # Compute the directory of this segment
            frame_dir = ''

            # Extract the corresponding RGBs
            # shape: T x H x W x C
            video_data = load_video_frames(frame_dir=frame_dir,
                                           start=start_frame,
                                           end=stop_frame,
                                           samp_rate=0.5)

            # TODO: Add padding so that all the segments have the same size?

            # Apply the data transformation, resizing and normalizing
            # Input has shape TxHxWxC
            # The transformation takes as input C x T x H x W, and returns
            video_data = transform(video_data.permute(3,0,1,2))  # Apply the transformation to the images (data augmentation)

            # Apply a pretrained I3D model (on Imagenet).
            # Apply a backbone so that the input data has shape CxTxHxW -> B x F' x T
            video_data = apply_backbone(torch.unsqueeze(video_data, 0), backbone)

            # Transpose to the final shape T x F
            video_data = rearrange(video_data, 'b f t -> b t f')

            data = {'data': video_data,
                    'verb_class': verb_class,
                    'noun_class': noun_class}

            # Save the preprocessed features into a .pt file
            segment_path = os.path.join(output_dir, narration_id + '.pt')
            torch.save(data, segment_path)


preprocess_dataset(data_path='',
                   label_path='',
                   output_dir='/data-local/data1-ssd/dpujolpe/Processed_EK100_I3D_frame')

