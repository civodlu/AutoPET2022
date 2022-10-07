import unittest
import trw
import torch
import numpy as np


class TestTransformsResizeModuloPadCrop(unittest.TestCase):
    def test_crop_mode_torch(self):
        batch = {
            'images': torch.rand([2, 3, 64, 64], dtype=torch.float32)
        }

        tfm = trw.transforms.TransformResizeModuloCropPad(60)
        transformed = tfm(batch)
        assert transformed['images'].shape == (2, 3, 60, 60)

    def test_crop_mode_torch_multiples(self):
        # test with multiple of `multiples_of` shape
        batch = {
            'images': torch.rand([2, 3, 64, 64], dtype=torch.float32)
        }

        tfm = trw.transforms.TransformResizeModuloCropPad(10)
        transformed = tfm(batch)
        assert transformed['images'].shape == (2, 3, 60, 60)

    def test_crop_mode_torch_different_shape(self):
        batch = {
            'images': torch.rand([2, 3, 64, 64], dtype=torch.float32),
            'images2': torch.rand([2, 1, 64, 64], dtype=torch.float32)
        }
        batch['images'][0, 0, 32, 32] = 42.0
        batch['images2'][0, 0, 32, 32] = 42.0

        tfm = trw.transforms.TransformResizeModuloCropPad(60)
        transformed = tfm(batch)

        # make sure we can handle different shapes of the same dimension
        assert transformed['images'].shape == (2, 3, 60, 60)
        assert transformed['images2'].shape == (2, 1, 60, 60)

        # make sure the crop/pad are the same for the different images
        indices = np.where(batch['images'].numpy() == 42)
        assert (batch['images2'][indices] == 42.0).all()

    def test_pad_mode_torch(self):
        batch = {
            'images': torch.rand([2, 3, 65, 65], dtype=torch.float32)
        }

        tfm = trw.transforms.TransformResizeModuloCropPad(32, mode='pad')
        transformed = tfm(batch)
        assert transformed['images'].shape == (2, 3, 96, 96)
