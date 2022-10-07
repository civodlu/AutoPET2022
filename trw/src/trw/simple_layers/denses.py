import numpy as np

from .simple_layers import SimpleModule
from ..layers import denses as denses_layers


def denses(parent, sizes, *args, **kwargs):
    full_sizes = [np.prod(parent.shape[1:])] + sizes
    module = denses_layers(sizes=full_sizes, *args, **kwargs, with_flatten=len(parent.shape) != 2)
    return SimpleModule(parent, module, shape=None)
