"""
The purpose to this module is to provide a convenient way to create static neural
network
"""

from .ordered_set import OrderedSet
from .simple_layers import SimpleOutputBase, SimpleMergeBase, SimpleLayerBase, SimpleModule
from .simple_layers_implementations import Input, OutputClassification, Flatten, Conv2d, ReLU, MaxPool2d, Linear, \
    ConcatChannels, OutputEmbedding, Conv3d, MaxPool3d, Reshape, BatchNorm2d, BatchNorm3d
from .compiled_net import compile_nn, find_layer_type, nodes_mark_output_dependencies, CompiledNet
from .denses import denses
from .convs import convs_3d, convs_2d
from .global_pooling import global_average_pooling_2d, global_average_pooling_3d, global_max_pooling_2d, \
    global_max_pooling_3d
from .shift_scale import ShiftScale
from .sub_tensor import SubTensor
