import torch.nn as nn
from ..layers import Flatten
from .simple_layers import SimpleModule


def global_average_pooling_2d(parent):
    assert len(parent.shape) == 4, 'expected a batch x channel x depth x height x width format'
    shape = [parent.shape[0], parent.shape[1]]
    
    # kernel must have the same size as the image to have `global`
    module = nn.AvgPool2d(kernel_size=[parent.shape[2], parent.shape[3]])
    module = nn.Sequential(module, Flatten())
    return SimpleModule(parent, module, shape=shape)


def global_max_pooling_2d(parent):
    assert len(parent.shape) == 4, 'expected a batch x channel x depth x height x width format'
    shape = [parent.shape[0], parent.shape[1]]
    
    # kernel must have the same size as the image to have `global`
    module = nn.MaxPool2d(kernel_size=[parent.shape[2], parent.shape[3]])
    module = nn.Sequential(module, Flatten())
    return SimpleModule(parent, module, shape=shape)


def global_average_pooling_3d(parent):
    assert len(parent.shape) == 5, 'expected a batch x channel x depth x height x width format'
    shape = [parent.shape[0], parent.shape[1]]
    
    # kernel must have the same size as the image to have `global`
    module = nn.AvgPool3d(kernel_size=[parent.shape[2], parent.shape[3], parent.shape[4]])
    module = nn.Sequential(module, Flatten())
    return SimpleModule(parent, module, shape=shape)


def global_max_pooling_3d(parent):
    assert len(parent.shape) == 5, 'expected a batch x channel x depth x height x width format'
    shape = [parent.shape[0], parent.shape[1]]
    
    # kernel must have the same size as the image to have `global`
    module = nn.MaxPool3d(kernel_size=[parent.shape[2], parent.shape[3], parent.shape[4]])
    module = nn.Sequential(module, Flatten())
    return SimpleModule(parent, module, shape=shape)
