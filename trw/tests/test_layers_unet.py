from unittest import TestCase
import trw
import torch


class TestUNet(TestCase):
    def test_unet_2d(self):
        """
        Simply test instantiation of a UNet-like model
        """
        for nb_blocs in [3, 4]:
            model = trw.layers.UNetBase(2, input_channels=3, output_channels=2, channels=[32 * n for n in range(1, nb_blocs + 1)])

            inputs = [
                torch.zeros([10, 3, 64, 32]),
                torch.zeros([10, 3, 32, 64]),
                torch.zeros([10, 3, 128, 128])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]

    def test_unet_no_latent(self):
        unet = trw.layers.UNetBase(2, 1, [8, 16, 32], 5)
        o = unet(torch.zeros([3, 1, 64, 64]))
        assert o.shape == (3, 5, 64, 64)

    def test_unet_with_latent(self):
        unet = trw.layers.UNetBase(2, 1, [8, 16, 32], 5, latent_channels=10)
        o = unet(torch.zeros([3, 1, 64, 64]), latent=torch.ones(3, 10, 1, 1))
        assert o.shape == (3, 5, 64, 64)

    def test_unet_3d(self):
        """
        Simply test instantiation of a UNet-like model
        """
        for nb_blocs in [3, 4]:
            model = trw.layers.UNetBase(3, input_channels=3, output_channels=2, channels=[16 * n for n in range(1, nb_blocs + 1)])

            inputs = [
                torch.zeros([4, 3, 16, 32, 16]),
                torch.zeros([4, 3, 16, 32, 16]),
                torch.zeros([4, 3, 16, 16, 32])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]

    def test_unet_kernels(self):
        nb_blocs = 3
        kernels = [3, 5, 7]

        for kernel in kernels:
            model = trw.layers.UNetBase(2, input_channels=1, output_channels=2, channels=[16 * n for n in range(1, nb_blocs + 1)], kernel_size=kernel)

            inputs = [
                torch.zeros([4, 1, 32, 64]),
                torch.zeros([4, 1, 32, 64]),
                torch.zeros([4, 1, 64, 32])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]

    def test_unet_strides(self):
        nb_blocs = 2
        strides = [2, 4]

        for stride in strides:
            kernel = 5
            model = trw.layers.UNetBase(2, input_channels=1, output_channels=2, channels=[16 * n for n in range(1, nb_blocs + 1)], kernel_size=kernel, strides=[stride] * nb_blocs)

            inputs = [
                torch.zeros([4, 1, 128, 128]),
                torch.zeros([4, 1, 64, 128]),
                torch.zeros([4, 1, 128, 64])
            ]

            for i in inputs:
                o = model(i)

                assert len(o.shape) == len(i.shape)
                assert o.shape[0] == i.shape[0]
                assert o.shape[1] == 2
                assert o.shape[2:] == i.shape[2:]
