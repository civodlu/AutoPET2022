from unittest import TestCase
import trw
import numpy as np
import skimage.filters
import torch
import skimage


class TestFilterGaussian(TestCase):
    def test_random_2d(self):
        """
        Compare gaussian filtering performed on a 2D image versus scipy implementation
        """
        sigmas = [1, 2, 3, 4, 5, 10]
        for sigma in sigmas:
            i = np.random.randint(0, 255, size=[128, 256, 3]).astype(np.float32)
            g = trw.train.FilterGaussian(input_channels=3, sigma=sigma, nb_dims=2)

            i_torch = np.reshape(i.transpose((2, 0, 1)), [1, 3, 128, 256])
            i_torch_filtered = g(torch.from_numpy(i_torch))
            i_torch_filtered = i_torch_filtered.numpy()[0].transpose((1, 2, 0))

            i_scipy = skimage.filters.gaussian(i, sigma=sigma, multichannel=True)

            max_error = np.max(i_torch_filtered[20:-20, 20:-20] - i_scipy[20:-20, 20:-20])
            self.assertTrue(max_error < 1.0)

    def test_random_3d(self):
        """
        Compare gaussian filtering performed on a 2D image versus scipy implementation
        """
        sigmas = [1]
        for sigma in sigmas:
            i = np.random.randint(0, 255, size=[96, 48, 64, 3]).astype(np.float32)
            g = trw.train.FilterGaussian(input_channels=3, sigma=sigma, nb_dims=3)

            i_torch = np.reshape(i.transpose((3, 0, 1, 2)), [1, 3, 96, 48, 64])
            i_torch_filtered = g(torch.from_numpy(i_torch))
            i_torch_filtered = i_torch_filtered.numpy()[0].transpose((1, 2, 3, 0))

            i_scipy = skimage.filters.gaussian(i, sigma=sigma, multichannel=True)

            max_error = np.max(i_torch_filtered[20:-20, 20:-20, 20:-20] - i_scipy[20:-20, 20:-20, 20:-20])
            self.assertTrue(max_error < 1.0)
