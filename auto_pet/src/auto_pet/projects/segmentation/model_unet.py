from functools import partial
from torch import nn
from trw.basic_typing import Batch
from trw.layers.unet_base import LatentConv
from trw.layers import BlockConvNormActivation
from typing import Dict
import trw
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossBinary(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = inputs.view((inputs.shape[0], -1))
        targets = targets.view((inputs.shape[0], -1))
        
        intersection = (inputs * targets).sum(dim=1)                            
        dice = (2.*intersection + smooth)/(inputs.sum(dim=1) + targets.sum(dim=1) + smooth)
        
        return 1 - dice


class ModelUNet(nn.Module):
    def __init__(self, filter_factor=2, kernel_middle_size=5) -> None:
        super().__init__()

        config = trw.layers.default_layer_config(
            norm_type=trw.layers.NormType.InstanceNorm,
            activation=nn.LeakyReLU
        )

        middle_block_fn = partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=kernel_middle_size))

        self.model = trw.layers.UNetBase(
            dim=3,
            input_channels=2,
            channels=[16 * filter_factor, 32 * filter_factor, 64 * filter_factor, 128 * filter_factor],
            output_channels=1,
            config=config,
            activation=None,
            middle_block_fn=middle_block_fn
        )

    def forward(self, batch: Batch) -> Dict:
        suv = batch['suv']
        ct = batch['ct']
        seg = batch['seg']
        assert len(ct.shape) == 5
        assert ct.shape[1] == 1

        features = torch.cat([ct, suv], dim=1)
        o = self.model(features)
        o = F.sigmoid(o)

        z_half = ct.shape[2] // 2 

        return {
            #'seg': trw.train.OutputSegmentationBinary(o, seg),
            'seg': trw.train.OutputSegmentationBinary(o, seg, criterion_fn=DiceLossBinary, output_postprocessing=lambda x: (x > 0.5).type(torch.long)),
            '2d_ct': trw.train.OutputEmbedding(ct[:, :, z_half]),
            '2d_suv': trw.train.OutputEmbedding(suv[:, :, z_half]),
            '2d_seg': trw.train.OutputEmbedding(seg[:, :, z_half]),
            '2d_found': trw.train.OutputEmbedding((o[:, :, z_half] > 0.5).type(torch.float32)),
        }

ModelUNetLargeFov = partial(ModelUNet, kernel_middle_size=9)