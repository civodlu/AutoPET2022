import torch
import torch.nn as nn
import functools

from .simple_layers import SimpleLayerBase, SimpleOutputBase, SimpleModule, SimpleMergeBase
from ..layers.utils import div_shape
from ..train.outputs_trw import OutputClassification as OutputClassification_train
from ..train.outputs_trw import OutputEmbedding as OutputEmbedding_train
from ..layers import Flatten as Flatten_layers
import numpy as np
import collections


class Input(SimpleLayerBase):
    """
    Represent an input (i.e., a feature) to a network
    """
    def __init__(self, shape: list, feature_name: str):
        """

        Args:
            shape: the shape of the input, including the batch size
            feature_name: the feature name to be used by the network from the batch
        """
        assert isinstance(shape, list)

        super().__init__(parents=None, shape=shape)
        self.feature_name = feature_name

    def get_module(self):
        # there is no module required for input
        return None


class OutputClassification(SimpleOutputBase):
    """
    Output class for classification
    """
    def __init__(self, node, output_name, classes_name, **kwargs):
        super().__init__(node=node, output_name=output_name, shape=node.shape)
        self.module_type = OutputClassification_train
        self.module_args = kwargs
        self.classes_name = classes_name

    def forward(self, inputs, batch):
        return self.module_type(inputs, batch[self.classes_name], classes_name=self.classes_name, **self.module_args)

    def get_module(self):
        # output layer, doesn't have a `Module` implementation
        return None
    
    
def return_output(outputs, batch):
    return outputs
    
    
class OutputEmbedding(SimpleOutputBase):
    """
    Create an embedding for display purposes
    """
    def __init__(self, node, output_name, functor=None):
        super().__init__(node=node, output_name=output_name, shape=node.shape)
        self.module_type = functools.partial(OutputEmbedding_train, functor=functor)

    def forward(self, inputs, batch):
        return self.module_type(inputs)

    def get_module(self):
        # output layer, doesn't have a `Module` implementation
        return None


class ReLU(SimpleModule):
    def __init__(self, node):
        super().__init__(node=node, module=nn.ReLU(), shape=node.shape)


class BatchNorm2d(SimpleModule):
    def __init__(self, node, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(
            node=node,
            module=nn.BatchNorm2d(num_features=node.shape[1], eps=eps, momentum=momentum, affine=affine),
            shape=node.shape
        )


class BatchNorm3d(SimpleModule):
    def __init__(self, node, eps=1e-5, momentum=0.1, affine=True):
        super().__init__(
            node=node,
            module=nn.BatchNorm3d(num_features=node.shape[1], eps=eps, momentum=momentum, affine=affine),
            shape=node.shape
        )


class _Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class Reshape(SimpleModule):
    """
    Reshape a tensor to another shape
    """
    def __init__(self, node, shape):
        reformated_shape = [s if s is not None else -1 for s in shape]
        super().__init__(node=node, module=_Reshape(shape=reformated_shape), shape=reformated_shape)


class Linear(SimpleModule):
    def __init__(self, node, out_features):
        assert len(node.shape) == 2, 'Linear input shape must be 2D, instead got={}'.format(node.shape)
        module_args = {'in_features': node.shape[1], 'out_features': out_features}
        super().__init__(node=node, module=nn.Linear(**module_args), shape=[node.shape[0], out_features])


class Flatten(SimpleModule):
    def __init__(self, node):
        super().__init__(node=node, module=Flatten_layers(), shape=[node.shape[0], np.prod(node.shape[1:])])


def _conv_2d_shape_fn(node, module_args):
    assert len(node.shape) == 4, 'must be `Batch size * Channels * Height * Width`'
    out_channels = module_args['out_channels']
    stride = module_args['stride']
    return [node.shape[0], out_channels] + div_shape(node.shape[2:], div=stride)


class Conv2d(SimpleModule):
    def __init__(self, node, out_channels, kernel_size, stride=1, padding='same'):
        module_args = {
            'in_channels': node.shape[1],
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride
        }

        if padding == 'same':
            module_args['padding'] = div_shape(kernel_size, 2)
        elif padding is None:
            pass
        else:
            assert 0, 'padding mode is not handled!'

        super().__init__(node=node, module=nn.Conv2d(**module_args), shape=_conv_2d_shape_fn(node=node, module_args=module_args))


def _conv_3d_shape_fn(node, module_args):
    assert len(node.shape) == 5, 'must be `Batch size * Channels * Depth * Height * Width`'
    out_channels = module_args['out_channels']
    stride = module_args['stride']
    return [node.shape[0], out_channels] + div_shape(node.shape[2:], div=stride)


class Conv3d(SimpleModule):
    def __init__(self, node, out_channels, kernel_size, stride=1, padding='same'):
        module_args = {
            'in_channels': node.shape[1],
            'out_channels': out_channels,
            'kernel_size': kernel_size,
            'stride': stride
        }

        if padding == 'same':
            module_args['padding'] = div_shape(kernel_size, 2)
        elif padding is None:
            pass
        else:
            assert 0, 'padding mode is not handled!'

        super().__init__(node=node, module=nn.Conv3d(**module_args), shape=_conv_3d_shape_fn(node=node, module_args=module_args))


class MaxPool2d(SimpleModule):
    def __init__(self, node, kernel_size, stride=None):
        module_args = {'kernel_size': kernel_size, 'stride': stride}
        super().__init__(node=node, module=nn.MaxPool2d(**module_args), shape=node.shape[0:2] + div_shape(node.shape[2:], 2))


class MaxPool3d(SimpleModule):
    def __init__(self, node, kernel_size, stride=None):
        module_args = {'kernel_size': kernel_size, 'stride': stride}
        super().__init__(node=node, module=nn.MaxPool3d(**module_args), shape=node.shape[0:2] + div_shape(node.shape[2:], 2))


class ConcatChannels(SimpleMergeBase):
    """
    Implement a channel concatenation layer
    """
    def __init__(self, nodes, flatten=False):
        assert isinstance(nodes, collections.Sequence), 'must be a list! Got={}'.format(type(nodes))
        super().__init__(parents=nodes, shape=ConcatChannels.calculate_shape(nodes))
        assert len(set(nodes)) == len(nodes), 'a node is duplicated! This is not handled!'
        self.flatten = flatten
        self.module = functools.partial(torch.cat, dim=1)

    @staticmethod
    def calculate_shape(parents):
        parent_shapes = []
        total_channels = 0
        for parent in parents:
            parent_shapes.append(parent.shape)

            expected_shape = parent_shapes[0][2:]
            found_shape = parent.shape[2:]
            assert expected_shape == found_shape, 'can\'t concate nodes as shape do not match. Expected={}, found={}'
            total_channels += parent.shape[1]

        shape = [None, total_channels] + parent_shapes[0][2:]
        return shape

    def get_module(self):
        return self.module
