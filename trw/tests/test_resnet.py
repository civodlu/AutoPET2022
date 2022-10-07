import unittest
import trw.layers
import torch


class TestResnet(unittest.TestCase):
    def test_resnet_construction(self):
        model = trw.layers.PreActResNet18(dimensionality=2, input_channels=3, output_channels=10)
        o = model(torch.zeros([5, 3, 32, 32]))
        assert o.shape == (5, 10)

        o_intermediates = model.forward_with_intermediate(torch.zeros([5, 3, 32, 32]))
        assert len(o_intermediates) == 5
        assert o_intermediates[0].shape == (5, 64, 32, 32)
        assert o_intermediates[1].shape == (5, 64, 32, 32)
        assert o_intermediates[2].shape == (5, 128, 16, 16)
        assert o_intermediates[3].shape == (5, 256, 8, 8)
        assert o_intermediates[4].shape == (5, 512, 4, 4)

        assert len(model.blocks) == 4
        assert len(model.blocks[0]) == 2
        assert len(model.blocks[1]) == 2
        assert len(model.blocks[2]) == 2
        assert len(model.blocks[3]) == 2

        assert model.blocks[0][0].conv1.ops[0].stride == (1, 1)
        assert model.blocks[0][0].conv2.ops[0].stride == (1, 1)

        assert model.blocks[1][0].conv1.ops[0].stride == (2, 2)
        assert model.blocks[1][0].conv2.ops[0].stride == (1, 1)


if __name__ == '__main__':
    unittest.main()
