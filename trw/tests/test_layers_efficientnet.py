from trw.layers import EfficientNet, default_layer_config
import torch
from unittest import TestCase

from trw.train.compatibility import Swish


class TestEfficientNet(TestCase):
    def test_efficient_net_construction(self):
        config = default_layer_config(dimensionality=2, activation=Swish)

        torch.random.manual_seed(0)
        i = torch.randn([4, 3, 224, 224])

        net = EfficientNet(dimensionality=2, input_channels=3, output_channels=10, config=config)
        o = net.feature_extractor(i)
        assert o.shape == (4, 1280, 7, 7)

        o = net(i)
        assert o.shape == (4, 10)

        intermediates = net.forward_with_intermediate(i)
        assert len(intermediates) == 9
