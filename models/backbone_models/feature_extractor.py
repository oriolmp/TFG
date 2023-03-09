# This script is based on the implementation from https://github.com/xlliu7/E2E-TAD/blob/24a1895d089601da0fc447e659fab63acfed61d4/util/misc.py#L298
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    # def cuda(self):
    #     tensors = self.tensors.cuda()
    #     mask = self.mask.cuda()
    #     return NestedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, encoder_config, model_name: str = 'I3D', fix_encoder:bool=False):
        super(FeatureExtractor, self).__init__()
        self.model_name: str = model_name
        self.encoder_config = encoder_config
        self.fix_encoder = fix_encoder
        self.use_upsample = encoder_config.temporal_upsample

        self._init_model()

    def _freeze_encoder(self):
        self.model.requires_grad_(False)

    def _init_model(self):
        if self.model_name == 'I3D':
            from Domain_adaptive_WTAL.src.models.backbone.models.pytorch_i3d import InceptionI3d
            self.model = InceptionI3d(400, in_channels=3).cuda()
            self.model.load_state_dict(torch.load('/home-net/dpujolpe/Domain_adaptive_WTAL/Domain_adaptive_WTAL/src/models/backbone/pretrained_models/I3D/rgb_imagenet.pt'))
            self.model.eval()
            self.num_channels = 1024
        elif self.model_name == 'slowfast':
            from Domain_adaptive_WTAL.src.models.backbone.models.slowfast import ResNet3dSlowFast
            self.model = ResNet3dSlowFast(None,
                                          depth=self.encoder_config.slowfast_depth,
                                          freeze_bn=self.encoder_config.freeze_bn,
                                          freeze_bn_affine=self.encoder_config.freeze_affine,
                                          slow_upsample=self.encoder_config.slow_upsample).cuda()
            self.num_channels = 2304
            self.model.load_pretrained_weight(None)
        elif self.model_name in ['tsm', 'tsn']:
            from Domain_adaptive_WTAL.src.models.backbone.models.tsm import TSM
            self.model = TSM(encoder_config=self.encoder_config,  # arch=self.encoder_config.tsm_base_model,
                             is_shift=self.model_name == 'tsm').cuda()
            self.num_channels = self.model.out_channels
        else:
            raise ValueError('The specified backbone pretrained_models is not currently implemented')

        # Freeze the backbone if necessary
        if self.fix_encoder:
            self._freeze_encoder()

    def _unfold(self, ip, kernel_size, stride):
        '''Expect NCTHW shaped tensor, extract sliding block for snippet-wise feature extraction'''
        # ip_ncts = rearrange(ip_ncthw, "n c t h w -> n c t (h w)")
        # ip_ncts = F.unfold(ip_ncts, (kernel_size, 1), stride=(stride, 1), padding=((kernel_size-stride)//2, 1))
        N, C, T, H, W = ip.shape
        pad_size = ((kernel_size - stride) // 2, (kernel_size - stride + 1) // 2)
        ip_pad = F.pad(ip, (0, 0, 0, 0, *pad_size), mode='constant', value=0)
        num_windows = T // stride
        start = torch.arange(num_windows).reshape([num_windows, 1]) * stride
        indices = (start + torch.arange(kernel_size)).view(-1)  # (num_windows, kernel_size)
        out = torch.index_select(ip_pad, dim=2, index=indices.to(ip.device))
        # pdb.set_trace()
        out = out.reshape(N, C, num_windows, kernel_size, H, W)
        out = rearrange(out, 'n c nw ks h w -> (n nw) c ks h w')
        return out

    def forward(self, tensor_list):
        '''tensor_list: tensors+mask'''
        '''
        if not isinstance(tensor_list, NestedTensor):
              b, t = tensor_list.shape[0], tensor_list.shape[2]
              mask = torch.zeros((b, t), dtype=torch.bool, device=tensor_list.device)
              tensor_list = NestedTensor(tensor_list, mask)
        '''

        tensors = tensor_list.tensors
        batch_size = tensors.shape[0]
        mask = tensor_list.mask
        shape = tensors.shape

        # it takes as input image sequence or feature vector sequence
        if len(shape) == 5:  # (n,c,t,h,w)
            pooler = F.adaptive_max_pool3d if self.encoder_config.spatial_pool == 'max' else F.adaptive_avg_pool3d

            if self.encoder_config.snippet_wise_feature:
                ip = self._unfold(tensor_list.tensors, self.encoder_config.snippet_length,
                                  self.encoder_config.snippet_stride)
                video_ft = self.model(ip).mean(2)  # (n*n_window, c, t, h, w)
                T = video_ft.shape[0] // batch_size
                video_ft_fold = video_ft.reshape(batch_size, T, *(video_ft.shape[1:]))  # (n, n_window, c, h, w)
                video_ft = video_ft_fold.transpose(1, 2)
            else:
                # fully convolutional feature extraction
                video_ft = self.model(tensor_list.tensors)  # [b,c,t, h, w]

            if isinstance(video_ft, (list, tuple)) and len(video_ft) == 1:
                video_ft = video_ft[0]

            if not isinstance(video_ft, (list, tuple)):
                if video_ft.ndim == 5:
                    video_ft = pooler(video_ft, [None, 1, 1])[..., 0, 0]  # extract_features
                if self.use_upsample:
                    video_ft = F.interpolate(video_ft, scale_factor=self.encoder_config.temporal_upscale, mode='linear')
                mask = F.interpolate(mask[None].float(), size=video_ft.shape[2], mode='nearest').to(torch.bool)[
                    0]  # [n, t]
                out = NestedTensor(video_ft, mask)
            else:
                # multilevel feature from backbone
                raise NotImplementedError

        return out
