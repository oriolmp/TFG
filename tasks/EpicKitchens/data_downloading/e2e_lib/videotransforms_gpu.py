import numbers
import random

import cv2
import numpy as np
import torch
import math
import torchvision.transforms as transforms
from typing import Tuple


class GroupResize(object):
    def __init__(self, new_size):
        assert isinstance(new_size, (list, tuple))
        self.new_size = new_size

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.new_size)

    def __call__(self, imgs):
        # imgs: shape CxTxHxW
        imgs = imgs.permute(1,0,2,3)    # TxCxHxW

        src_shape = imgs[0].shape[-2:]
        new_h, new_w = 0, 0
        if src_shape[0] < src_shape[1]:
            new_h = min(self.new_size)
            new_w = max(self.new_size)
        else:
            new_h = max(self.new_size)
            new_w = min(self.new_size)

        new_size = (new_w, new_h)

        res = torch.nn.functional.interpolate(imgs, size=new_size, mode="bilinear").permute(1,0,2,3)

        res /= 255.

        return res



class GroupResizeShorterSide(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, imgs):
        # imgs: shape CxTxHxW
        imgs = imgs.permute(1, 0, 2, 3)  # TxCxHxW

        shorter = min(imgs[0].shape[-2:])
        sc = self.new_size / shorter

        res = torch.nn.functional.interpolate(imgs, size=(int(imgs.shape[1] * sc), int(imgs.shape[2] * sc)), mode="bilinear").permute(1,0,2,3)

        res /= 255.

        return res

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.new_size)


class GroupNormalize(object):
    """Normalize images.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert images from BGR to RGB,
            default is true.
    """

    def __init__(self, to_rgb=False):
        self.to_rgb = to_rgb

    def __call__(self, imgs):
        """Call function to normalize images.

        Args:
            imgs (dict): CxTxHxW tensor

        Returns:
            processed imgs
        """
        if self.to_rgb:
            imgs[[0,2], ...] = imgs[[2,0], ...] # Reorder from BGR to RGB

        # Normalize from 0-1 to -1,+1 (for the I3D model)
        imgs = imgs*2 - 1

        return imgs


if __name__ == '__main__':
    pass
