from unittest import TestCase
import trw.transforms
import torch
import numpy as np


class TestRenormalize(TestCase):
    def test_renormalize_torch(self):
        data = np.random.normal(25, 4, size=(1000, 4, 4))
        data = torch.from_numpy(data)
        data_transformed = trw.transforms.renormalize(data, desired_mean=2, desired_std=3)

        mean = torch.mean(data_transformed)
        std = torch.std(data_transformed)
        assert abs(mean - 2) < 1e-5
        assert abs(std - 3) < 1e-5

    def test_renormalize_numpy(self):
        data = np.random.normal(25, 4, size=(1000, 4, 4))
        data_transformed = trw.transforms.renormalize(data, desired_mean=2, desired_std=3)

        mean = np.mean(data_transformed)
        std = np.std(data_transformed)
        assert abs(mean - 2) < 1e-5
        assert abs(std - 3) < 1e-5
