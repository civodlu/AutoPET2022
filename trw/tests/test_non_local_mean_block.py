
import unittest
import trw
from trw.layers import BlockNonLocal, linear_embedding
import torch


class TestNonLocalMean(unittest.TestCase):
    def test_block(self):
        config = trw.layers.default_layer_config(dimensionality=2)
        block = BlockNonLocal(
            config=config, 
            input_channels=3,
            intermediate_channels=6,
            f_mapping_fn=linear_embedding,
            g_mapping_fn=linear_embedding,
        )
        
        o = block(torch.zeros(2, 3, 16, 16))
        assert o.shape == (2, 3, 16, 16)


if __name__ == '__main__':
    unittest.main()
