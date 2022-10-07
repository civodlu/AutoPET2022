from unittest import TestCase
import trw
import numpy as np
import torch


class TestSimpleLayers(TestCase):
    def test_conv2d(self):
        # make sure we the syntatic sugar for the arguments are correctly managed (e.g., kernel sizes are expanded...)
        i = trw.simple_layers.Input([None, 3, 28, 28], feature_name='input_2d_rgb')
        o = trw.simple_layers.convs_2d(
            i,
            channels=[16, 32],
            convolution_kernels=5,
            strides=1,
            pooling_size=2,
            convolution_repeats=1,
            with_flatten=True,
            dropout_probability=0.5,
            padding='same')

        net = trw.simple_layers.compile_nn([o])
        r = net({'input_2d_rgb': torch.zeros([10, 3, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (10, 32 * (28 // 2 // 2) ** 2)

    def test_conv3d(self):
        # make sure we the syntatic sugar for the arguments are correctly managed (e.g., kernel sizes are expanded...)
        i = trw.simple_layers.Input([None, 3, 28, 28, 28], feature_name='input_3d_rgb')
        o = trw.simple_layers.convs_3d(
            i,
            channels=[10, 11],
            convolution_kernels=3,
            strides=1,
            pooling_size=2,
            convolution_repeats=1,
            with_flatten=True,
            dropout_probability=0.5,
            padding='same')

        net = trw.simple_layers.compile_nn([o])
        r = net({'input_3d_rgb': torch.zeros([5, 3, 28, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 11 * (28 // 2 // 2) ** 3)

    def test_denses(self):
        i = trw.simple_layers.Input([None, 3], feature_name='input')
        o = trw.simple_layers.denses(
            i,
            sizes=[16, 32],
            dropout_probability=0.5,
            activation=torch.nn.ReLU6,
            last_layer_is_output=True)
        net = trw.simple_layers.compile_nn([o])
        r = net({'input': torch.zeros([5, 3], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 32)

    def test_shift_scale(self):
        i = trw.simple_layers.Input([None, 3, 28, 28], feature_name='input')
        o = trw.simple_layers.ShiftScale(i, mean=10.0, standard_deviation=5.0)

        net = trw.simple_layers.compile_nn([o])
        r = net({'input': torch.zeros([5, 3, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 3, 28, 28)
        assert torch.abs(torch.mean(r[o]) - ((0.0 - 10.0) / 5.0)) < 1e-5

    def test_global_average_pooling_2d(self):
        i = trw.simple_layers.Input([None, 3, 28, 28], feature_name='input')
        o = trw.simple_layers.global_average_pooling_2d(i)

        net = trw.simple_layers.compile_nn([o])
        r = net({'input': torch.zeros([5, 3, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 3)

    def test_global_max_pooling_2d(self):
        i = trw.simple_layers.Input([None, 3, 28, 28], feature_name='input')
        o = trw.simple_layers.global_max_pooling_2d(i)

        net = trw.simple_layers.compile_nn([o])
        r = net({'input': torch.zeros([5, 3, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 3)

    def test_global_max_pooling_3d(self):
        i = trw.simple_layers.Input([None, 3, 4, 28, 28], feature_name='input')
        o = trw.simple_layers.global_max_pooling_3d(i)

        net = trw.simple_layers.compile_nn([o])
        r = net({'input': torch.zeros([5, 3, 4, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 3)

    def test_global_average_pooling_3d(self):
        i = trw.simple_layers.Input([None, 3, 4, 28, 28], feature_name='input')
        o = trw.simple_layers.global_average_pooling_3d(i)

        net = trw.simple_layers.compile_nn([o])
        r = net({'input': torch.zeros([5, 3, 4, 28, 28], dtype=torch.float32)})
        assert len(r) == 1
        assert r[o].shape == (5, 3)

    def test_output_classification(self):
        i = trw.simple_layers.Input([None, 3, 6, 6], feature_name='input')
        i = trw.simple_layers.denses(i, [4, 2])
        o = trw.simple_layers.OutputClassification(i, output_name='classification', classes_name='output')
        net = trw.simple_layers.compile_nn([o])

        r = net({
            'input': torch.zeros([5, 3, 6, 6], dtype=torch.float32),
            'output': torch.zeros([5], dtype=torch.int64)
        })
        assert len(r) == 1
        assert r['classification'].output.shape == (5, 2)

    def test_output_embedding(self):
        i = trw.simple_layers.Input([None, 3, 6, 6], feature_name='input')
        o = trw.simple_layers.OutputEmbedding(i, output_name='embedding')
        net = trw.simple_layers.compile_nn([o])

        i_torch = torch.randn([5, 32]).float()
        r = net({'input': i_torch})
        assert len(r) == 1
        assert (r['embedding'].output == i_torch).all()

    def test_batchnorm2d(self):
        i = trw.simple_layers.Input([None, 3, 32, 32], feature_name='input')
        o = trw.simple_layers.BatchNorm2d(i)
        net = trw.simple_layers.compile_nn([o])

        i_torch = torch.randn([5, 3, 32, 32]).float()
        r = net({'input': i_torch})
        assert len(r) == 1
        assert r[o].shape == (5, 3, 32, 32)

    def test_batchnorm3d(self):
        i = trw.simple_layers.Input([None, 3, 32, 32, 32], feature_name='input')
        o = trw.simple_layers.BatchNorm3d(i)
        net = trw.simple_layers.compile_nn([o])

        i_torch = torch.randn([5, 3, 32, 32, 32]).float()
        r = net({'input': i_torch})
        assert len(r) == 1
        assert r[o].shape == (5, 3, 32, 32, 32)

    def test_basic_2d(self):
        i = trw.simple_layers.Input([None, 3, 6, 6], feature_name='input')
        i = trw.simple_layers.Conv2d(i, out_channels=16, kernel_size=5, stride=1, padding='same')
        i = trw.simple_layers.ReLU(i)
        i = trw.simple_layers.MaxPool2d(i, kernel_size=2)
        i2 = trw.simple_layers.ReLU(i)
        i = trw.simple_layers.ConcatChannels([i, i2])
        i = trw.simple_layers.Flatten(i)
        o = trw.simple_layers.Linear(i, 8)
        net = trw.simple_layers.compile_nn([o])

        r = net({'input': torch.randn([5, 3, 6, 6]).float()})
        assert len(r) == 1
        assert r[o].shape == (5, 8)

    def test_basic_3d(self):
        i = trw.simple_layers.Input([None, 3, 6, 6, 6], feature_name='input')
        i = trw.simple_layers.Conv3d(i, out_channels=16, kernel_size=5, stride=1, padding='same')
        i = trw.simple_layers.ReLU(i)
        o = trw.simple_layers.MaxPool3d(i, kernel_size=2)
        net = trw.simple_layers.compile_nn([o])

        r = net({'input': torch.randn([5, 3, 6, 6, 6]).float()})
        assert len(r) == 1
        assert r[o].shape == (5, 16, 3, 3, 3)

    def test_sub_tensor(self):
        n = trw.simple_layers.Input([None, 1, 32, 32], feature_name='input')
        n = trw.simple_layers.SubTensor(n, [0, 10, 15], [1, 14, 22])
        net = trw.simple_layers.compile_nn([n])

        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)
        o = net({'input': i})

        assert o[n].shape == (5, 1, 4, 7)
        assert (o[n] == i[:, 0:1, 10:14, 15:22]).all()

    def test_sub_tensor_partial(self):
        n = trw.simple_layers.Input([None, 3, 32, 32], feature_name='input')
        n = trw.simple_layers.SubTensor(n, [0], [1])
        net = trw.simple_layers.compile_nn([n])

        i = torch.randn([5, 3, 32, 32], dtype=torch.float32)
        o = net({'input': i})

        assert o[n].shape == (5, 1, 32, 32)
        assert (o[n] == i[:, 0:1]).all()
