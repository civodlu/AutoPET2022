"""
Notes:

    * we can't use local functions or lambda to simplify the pickling and unpickling
"""

import weakref
import torch.nn as nn
import torch
from ..train import get_device
from .ordered_set import OrderedSet


class SimpleLayerBase:
    """
    Base layer for our simplified network specification

    Record the network node by node and keep track of the important information: parents, children, size.

    Note:

        * `nn.Module` must be created during the initialization. This is to make sure we can easily share the
            network for different sub-models
    """
    def __init__(self, parents, shape):
        # we REQUIRE an ordered set as this will define the evaluation order of the parents/children
        self.children = OrderedSet()
        self.parents = OrderedSet()
        self.shape = shape

        if parents is not None:
            for parent in parents:
                if parent is not None:
                    # the parent has the current node as child
                    # the parent manage the storage of the children to avoid
                    # memory leaks (self referencing)
                    parent.children.add(weakref.ref(self))
    
                    # the child has parent `parent`. We need to use weak reference
                    # to avoid self-referenced memory leaks
                    self.parents.add(parent)

    def get_module(self):
        """
        Return a `nn.Module`
        """
        assert 0, 'implement in the derived classes!'


class SimpleMergeBase(SimpleLayerBase):
    """
    Base class for nodes with multiple inputs
    """


class SimpleOutputBase(SimpleLayerBase):
    """
    Base class to calculate an output
    """
    def __init__(self, node, output_name, shape):
        super().__init__(parents=[node], shape=shape)
        self.output_name = output_name

    def forward(self, inputs, batch):
        """
        Create a `trw.train.Output` from the inputs

        Args:
            inputs: a list of inputs of the output node
            batch: the batch of data fed to the network

        Returns:
            a `trw.train.Output` object

        """
        assert 0, 'override this method in derived classes. Must create a `trw.train.Output`'


class SimpleModule(SimpleLayerBase):
    """
    Generic module

    Module must have a single input and all the module's parameters should be on the same device.
    """
    def __init__(self, node, module, shape=None):
        """

        Args:
            node: the parent node
            module: the module to be used to perform the forward calculation. Acceptable type are `torch.nn.Module`, in that case
                the module may have trainable parameters or a functional layer but may NOT have trainable parameters
            shape: if None, the size will be calculated with a fake input based on the parents' shape. if
                specified, use this shape as the shape of this node
        """
        parents = [node]
        super().__init__(parents=parents, shape=SimpleModule.calculate_shape(shape, module, parents))

        assert isinstance(module, nn.Module), 'module must be a `torch.nn.Module`'
        self.module = module

    @staticmethod
    def calculate_shape(shape, module, parents):
        if shape is not None:
            return shape

        # the shape is not deterministically known, so try to run a single example through the module
        # and use its shape as the calculated shape
        input_shape = next(iter(parents)).shape.copy()
        nb_samples = input_shape[0]
        input_shape[0] = 2
        device = get_device(module)
        input_data = torch.zeros(input_shape).to(device)
        output = module(input_data)

        output_shape = list(output.shape)
        output_shape[0] = nb_samples
        return output_shape

    def get_module(self):
        return self.module
