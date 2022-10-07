from .simple_layers import SimpleModule
from ..layers import convs_2d as conv2d_layers
from ..layers import convs_3d as conv3d_layers


def convs_3d(parent, channels, *args, **kwargs):
    module = conv3d_layers(input_channels=parent.shape[1], channels=channels, *args, **kwargs)
    return SimpleModule(parent, module, shape=None)


def convs_2d(parent, channels, *args, **kwargs):
    module = conv2d_layers(input_channels=parent.shape[1], channels=channels, *args, **kwargs)
    return SimpleModule(parent, module, shape=None)
