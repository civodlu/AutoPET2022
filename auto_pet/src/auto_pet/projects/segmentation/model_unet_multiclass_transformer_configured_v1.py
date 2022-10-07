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

from .model_unet_multiclass_deepsupervision import ModelUNetMulticlassDeepSupervision, LossDiceCrossEntropyFocal2
from .model_unet_transformer import UNETR


UNetTransformer_lung = partial(
    ModelUNetMulticlassDeepSupervision,
    model=UNETR(
        in_channels=3,
        out_channels=2,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        conv_block=True,
        dropout_rate=0.0,
    ),
    with_deep_supervision=False,
    loss_fn=LossDiceCrossEntropyFocal2,
    with_ct_lung=True,
    input_target_shape=(3, 96, 96, 96)
)