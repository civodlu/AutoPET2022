from unittest import TestCase
import trw.train
import numpy as np
import torch
import trw.utils


class TestUpsample(TestCase):
    def test_nearest(self):
        # test resmapling of a 2D array with 2 samples and 1 filter per sample
        a = np.asarray(
            [
                [[[1, 2, 3],
                [4, 5, 6]]],

                [[[7, 8, 9],
                 [10, 11, 12]]],
            ],
            dtype=np.float32)

        upsampled_a = trw.utils.upsample(torch.from_numpy(a), size=[4, 6], mode='nearest')
        assert upsampled_a.shape == (2, 1, 4, 6)
        assert upsampled_a[0, 0, 0, 0] == 1
        assert upsampled_a[0, 0, 1, 1] == 1
        assert upsampled_a[0, 0, 0, 2] == 2
        assert upsampled_a[0, 0, 1, 2] == 2
        assert upsampled_a[0, 0, 3, 4] == 6
        assert upsampled_a[0, 0, 3, 5] == 6

        assert upsampled_a[1, 0, 0, 2] == 8
        assert upsampled_a[1, 0, 1, 2] == 8
        assert upsampled_a[1, 0, 0, 4] == 9
        assert upsampled_a[1, 0, 1, 5] == 9
        assert upsampled_a[1, 0, 3, 4] == 12
        assert upsampled_a[1, 0, 3, 5] == 12

    def test_linear(self):
        # test resmapling of a 2D array with 2 samples and 1 filter per sample
        a = np.asarray(
            [
                [[[0, 1],
                [2, 3]]],

                [[[40, 41],
                 [42, 43]]],
            ],
            dtype=np.float32)

        upsampled_a = trw.utils.upsample(torch.from_numpy(a), size=[5, 5], mode='linear')
        assert upsampled_a.shape == (2, 1, 5, 5)
        assert upsampled_a[0, 0, 0, 0] == 0
        assert upsampled_a[0, 0, 0, 2] == 0.5
        assert upsampled_a[0, 0, 0, 4] == 1.0
        assert upsampled_a[0, 0, 4, 4] == 3
        assert upsampled_a[0, 0, 4, 2] == 2.5
        assert upsampled_a[0, 0, 2, 2] == (0 + 1 + 2 + 3) / 4.0

        assert upsampled_a[1, 0, 2, 2] == (40 + 41 + 42 + 43) / 4.0

    def test_upsample_int_2d(self):
        # test resmapling of a 2D array with 2 samples and 1 filter per sample
        a = np.asarray(
            [
                [[[1, 2, 3],
                [4, 5, 6]]],

                [[[7, 8, 9],
                 [10, 11, 12]]],
            ],
            dtype=np.int)

        upsampled_a = trw.utils.upsample(torch.from_numpy(a), size=[4, 6])
        assert upsampled_a.shape == (2, 1, 4, 6)
        assert upsampled_a[0, 0, 0, 0] == 1
        assert upsampled_a[0, 0, 1, 1] == 1
        assert upsampled_a[0, 0, 0, 2] == 2
        assert upsampled_a[0, 0, 1, 2] == 2
        assert upsampled_a[0, 0, 3, 4] == 6
        assert upsampled_a[0, 0, 3, 5] == 6

        assert upsampled_a[1, 0, 0, 2] == 8
        assert upsampled_a[1, 0, 1, 2] == 8
        assert upsampled_a[1, 0, 0, 4] == 9
        assert upsampled_a[1, 0, 1, 5] == 9
        assert upsampled_a[1, 0, 3, 4] == 12
        assert upsampled_a[1, 0, 3, 5] == 12

    def test_upsample_int_1d(self):
        a = np.asarray(
            [[[1, 2]]], dtype=np.int)

        upsampled_a = trw.utils.upsample(torch.from_numpy(a), size=[4])
        assert upsampled_a.shape == (1, 1, 4)

        assert upsampled_a[0, 0, 0] == 1
        assert upsampled_a[0, 0, 1] == 1
        assert upsampled_a[0, 0, 2] == 2
        assert upsampled_a[0, 0, 3] == 2

    def test_upsample_int_3d(self):
        # test resmapling of a 2D array with 2 samples and 1 filter per sample
        a = np.asarray(
            [[
                [[[1, 2],
                 [4, 5]],

                 [[6, 7],
                  [8, 9]]],
            ]],
            dtype=np.int)

        upsampled_a = trw.utils.upsample(torch.from_numpy(a), size=[2, 4, 6])
        assert upsampled_a.shape == (1, 1, 2, 4, 6)

        assert upsampled_a[0, 0, 1, 0, 0] == 6

        assert upsampled_a[0, 0, 0, 0, 0] == 1
        assert upsampled_a[0, 0, 0, 0, 1] == 1
        assert upsampled_a[0, 0, 0, 0, 2] == 1

        assert upsampled_a[0, 0, 0, 0, 3] == 2
        assert upsampled_a[0, 0, 0, 0, 4] == 2
        assert upsampled_a[0, 0, 0, 0, 5] == 2

        assert upsampled_a[0, 0, 0, 1, 0] == 1
        assert upsampled_a[0, 0, 0, 1, 1] == 1
        assert upsampled_a[0, 0, 0, 1, 2] == 1

        assert upsampled_a[0, 0, 0, 1, 3] == 2
        assert upsampled_a[0, 0, 0, 1, 4] == 2
        assert upsampled_a[0, 0, 0, 1, 5] == 2

        assert upsampled_a[0, 0, 0, 2, 0] == 4
        assert upsampled_a[0, 0, 0, 2, 1] == 4
        assert upsampled_a[0, 0, 0, 2, 2] == 4

        assert upsampled_a[0, 0, 0, 2, 3] == 5
        assert upsampled_a[0, 0, 0, 2, 4] == 5
        assert upsampled_a[0, 0, 0, 2, 5] == 5

        # other slice
        assert upsampled_a[0, 0, 1, 0, 0] == 6
        assert upsampled_a[0, 0, 1, 0, 1] == 6
        assert upsampled_a[0, 0, 1, 0, 2] == 6

        assert upsampled_a[0, 0, 1, 0, 3] == 7
        assert upsampled_a[0, 0, 1, 0, 4] == 7
        assert upsampled_a[0, 0, 1, 0, 5] == 7

        assert upsampled_a[0, 0, 1, 1, 0] == 6
        assert upsampled_a[0, 0, 1, 1, 1] == 6
        assert upsampled_a[0, 0, 1, 1, 2] == 6

        assert upsampled_a[0, 0, 1, 1, 3] == 7
        assert upsampled_a[0, 0, 1, 1, 4] == 7
        assert upsampled_a[0, 0, 1, 1, 5] == 7

        assert upsampled_a[0, 0, 1, 2, 0] == 8
        assert upsampled_a[0, 0, 1, 2, 1] == 8
        assert upsampled_a[0, 0, 1, 2, 2] == 8

        assert upsampled_a[0, 0, 1, 2, 3] == 9
        assert upsampled_a[0, 0, 1, 2, 4] == 9
        assert upsampled_a[0, 0, 1, 2, 5] == 9

        assert upsampled_a[0, 0, 1, 3, 0] == 8
        assert upsampled_a[0, 0, 1, 3, 1] == 8
        assert upsampled_a[0, 0, 1, 3, 2] == 8

        assert upsampled_a[0, 0, 1, 3, 3] == 9
        assert upsampled_a[0, 0, 1, 3, 4] == 9
        assert upsampled_a[0, 0, 1, 3, 5] == 9







