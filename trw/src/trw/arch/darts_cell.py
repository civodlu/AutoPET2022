import torch.nn as nn
import torch
from . import darts_ops
import torch.nn.functional as F
import collections
import numpy as np


class MixedLayer(nn.Module):
    """
    Represents a mixture of weighted primitive units
    """
    def __init__(self, primitives, c, stride):

        super().__init__()
        self.layers = nn.ModuleList()
        self.primitive_names = list(primitives.keys())

        for primitive_name, primitive_fn in primitives.items():
            layer = primitive_fn(c, stride, False)
            # layer = nn.Sequential(layer, nn.BatchNorm2d(c, affine=False))  # TODO deviation here
            self.layers.append(layer)

    def forward(self, x, weights):
        assert len(weights) == len(self.layers), 'we have {} layers. Expecting same number of weights but got={}'.format(len(self.layers), len(len(weights)))

        if len(weights) == 1:
            # we have a single weight, there is no need to apply the weighting
            # since there is no primitive choice!
            res = self.layers[0](x)
        else:
            # weighted sum of the primitives
            res = [w * layer(x) for w, layer in zip(weights, self.layers)]
            res = sum(res)
        return res


def default_cell_output(node_outputs, nb_outputs_to_use=4):
    return torch.cat(node_outputs[-nb_outputs_to_use:], dim=1)


class SpecialParameter(nn.Parameter):
    """
    Tag a parameter as special such as `DARTS` parameter. These should be handled differently
    depending on the phase: training the DARTS cell parameters or the weight parameters
    """


def _identity(x):
    return x


class Cell(nn.Module):
    def __init__(self, primitives, cpp, cp, c, is_reduction, is_reduction_prev, internal_nodes=4, cell_merge_output_fn=default_cell_output, weights=None, with_preprocessing=True, genotype=None):
        """
        :param internal_nodes: number of nodes inside a cell
        :param cpp: the previous's previous number of channels
        :param cp: the previous number of channels
        :param c: the current number of channels
        :param is_reduction: indicates whether to reduce the output maps width
        :param cell_merge_output_fn: defines how to calculate the cell output from the node outputs
        :param is_reduction_prev: when previous cell reduced width, s1_d = s0_d//2
        in order to keep same shape between s1 and s0, we adopt prep0 layer to
        reduce the s0 width by half.
        :param genotype: a list of primitive name. This describe the primitive to be used for each internal node link
        """
        super().__init__()

        assert isinstance(primitives, collections.OrderedDict), 'MUST have an ordered dict so that we can easily map weights to a primitive'

        # indicating current cell is is_reduction or not
        self.is_reduction = is_reduction
        self.is_reduction_prev = is_reduction_prev
        self.genotype = genotype

        # preprocess0 deal with output from prev_prev cell
        if with_preprocessing:
            # TODO remove this preprocessing... we should not assume 2D images!
            if is_reduction_prev:
                # if prev cell has reduced channel/double width,
                # it will reduce width by half
                # TODO do we really need this branch?
                self.preprocess0 = darts_ops.ReduceChannels2d(cpp, c, affine=False)
            else:
                self.preprocess0 = darts_ops.ReLUConvBN2d(cpp, c, 1, 1, 0, affine=False)

            # preprocess1 deal with output from prev cell
            self.preprocess1 = darts_ops.ReLUConvBN2d(cp, c, 1, 1, 0, affine=False)

        else:
            self.preprocess0 = _identity
            self.preprocess1 = _identity

        # internal_nodes inside a cell
        self.internal_nodes = internal_nodes
        self.cell_merge_output_fn = cell_merge_output_fn

        self.layers = nn.ModuleList()

        self.nb_links = 0
        for i in range(self.internal_nodes):
            # for each i inside cell, it connects with all previous output
            # plus previous two cells' output
            for j in range(2 + i):
                # for is_reduction cell, it will reduce the heading 2 inputs only
                stride = 2 if is_reduction and j < 2 else 1

                if genotype is None:
                    # here we are in search mode
                    layer = MixedLayer(primitives, c, stride)
                else:
                    # here we already have a specified configuration so just construct
                    # the internal links with the specified nodes
                    primitive_name = genotype[self.nb_links]
                    assert primitive_name in primitives, 'can\'t find primitive `{}` among our primitives={}'.format(primitive_name, list(primitives.keys()))
                    p = primitives[genotype[self.nb_links]]
                    layer = MixedLayer({primitive_name: p}, c, stride)

                self.layers.append(layer)
                self.nb_links += 1

        if genotype is None:
            self.weights = self._create_weights(primitives, weights=weights)
        else:
            # create fake weights: these will not be used since we have a single
            # primitive per link
            self.weights = self._create_weights([1], weights=weights)
            with torch.no_grad():
                self.weights.fill_(1.0)

    def _create_weights(self, primitives, weights):
        """
        Create the weights. Do not store them directly in the model parameters else they will be optimized
        too!
        """
        if weights is None:
            weights = SpecialParameter(torch.randn(self.nb_links, len(primitives)))
            with torch.no_grad():
                weights.mul_(1e-3)
        else:
            assert len(weights.shape) == 2
            assert weights.shape[0] == self.nb_links, 'expected total number of cell links={} but given={}'.format(self.nb_links, weights.shape[0])
            assert weights.shape[1] == len(primitives), 'expected total number of cell links={} but given={}'.format(len(primitives), weights.shape[1])
        return weights

    def forward(self, parents):
        assert len(parents) == 2
        s0 = parents[0]
        s1 = parents[1]

        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        normalized_weights = F.softmax(self.weights, dim=-1)

        # for each node, receive input from all previous intermediate nodes and s0, s1
        for i in range(self.internal_nodes):
            # [40, 16, 32, 32]
            s = sum(self.layers[offset + j](h, normalized_weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            # append one state since s is the elem-wise addition of all output
            states.append(s)
            # print('node:',i, s.shape, self.is_reduction)

        # concat along dim=channel
        return self.cell_merge_output_fn(states)

    def get_weights(self):
        """
        Returns:
            The primitive weights for this cell. This is useful if we want to share the weights
            among multiple cells
        """
        return self.weights

    def get_genotype(self):
        """
        Returns:
            The `genotype` of the cell given the current primitive weighting
        """
        best_primitive_per_internal_link = []
        weights_np = self.weights.detach().cpu().numpy()
        assert len(self.layers) == len(weights_np)
        for link_id, link_weights in enumerate(weights_np):
            best_primitive = int(np.argmax(link_weights))
            layer = self.layers[link_id]
            primitive_name = layer.primitive_names[best_primitive]
            best_primitive_per_internal_link.append(primitive_name)
        return best_primitive_per_internal_link
