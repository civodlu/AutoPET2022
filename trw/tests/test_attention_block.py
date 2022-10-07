
import unittest
import trw
import torch

        
class TestAttentionBlock(unittest.TestCase):
    def test_block(self):
        config = trw.layers.default_layer_config(dimensionality=3)
        block = trw.layers.unet_attention.BlockAttention(
            config=config, 
            input_channels=2,
            gating_channels=3,
            intermediate_channels=4)

        x = torch.zeros([4, 2, 4, 5, 6])
        g = torch.zeros([4, 3, 4, 5, 6])
        o = block(g, x)
        assert o.shape == (4, 2, 4, 5, 6)

    def test_attention_unet(self):
        config = trw.layers.default_layer_config()
        model = trw.layers.UNetAttention(
            dim=2,
            input_channels=2,
            channels=[3, 4, 5],
            output_channels=1,
            config=config
        )

        o = model(torch.zeros(3, 2, 16, 16))
        assert o.shape == (3, 1, 16, 16)


if __name__ == '__main__':
    unittest.main()
