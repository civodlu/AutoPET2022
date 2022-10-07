from functools import partial
from timeit import repeat
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

from .model_unet_multiclass import ModelUNetMulticlass, LossDiceCrossEntropyFocal
from .model_unet_multiclass_deepsupervision import ModelUNetMulticlassDeepSupervision, LossDiceCrossEntropyFocal2

from trw.layers.blocks import BlockConvNormActivation, BlockUpDeconvSkipConv
from trw.layers import LayerConfig, default_layer_config
from trw.layers.unet_base import BlockConvType, BlockTypeConvSkip, Up


SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary_large = partial(
    ModelUNetMulticlassDeepSupervision,
    model=trw.layers.UNetBase(
        dim=3,
        input_channels=5,
        channels=[32, 48, 64, 96, 128, 156],
        output_channels=2,
        config=trw.layers.default_layer_config(
            norm_type=trw.layers.NormType.InstanceNorm,
            activation=nn.LeakyReLU,
            norm_kwargs={'affine': True},
        ),
        activation=None,
        middle_block_fn=partial(LatentConv, block=partial(BlockConvNormActivation, kernel_size=5))
    ),
    with_deep_supervision=True,
    boundary_loss_factor=1.0,
    loss_fn=partial(LossDiceCrossEntropyFocal2, ce_factor=0.5, gamma=1.0),
    with_ct_lung=True,
    with_ct_soft=True,
    with_pet_hot=True,
    input_target_shape=(5, 256, 256, 256)
)
