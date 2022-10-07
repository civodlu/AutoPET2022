import trw
from unittest import TestCase
import torch


class TestCallbackExportConvolutionKernel(TestCase):
    def test_simple(self):
        n = trw.simple_layers.Input([None, 1, 16, 17, 18], feature_name='image')

        convolution_kernels = [(7, 7, 7), (5, 5, 5), (3, 3, 3)]
        n = trw.simple_layers.convs_3d(
            n,
            convolution_kernels=convolution_kernels,
            strides=1,
            pooling_size=2,
            convolution_repeats=1,
            with_flatten=True,
            dropout_probability=0.5,
            channels=[4, 8, 16]
        )
        n = trw.simple_layers.OutputEmbedding(n, 'embedding')
        net = trw.simple_layers.compile_nn(output_nodes=[n])

        batch = {
            'image': torch.ones([10, 1, 16, 17, 18], dtype=torch.float32)
        }

        dataset = {
            'dataset': {
                'train': trw.train.SequenceArray(batch)
            }
        }

        o = net(batch)
        output_values = o['embedding'].output
        assert output_values.shape == (10, 128)

        options = trw.train.Options()
        callback = trw.callbacks.CallbackExportConvolutionKernel(
            find_convolution_fn=trw.train.find_last_forward_convolution)

        callback(options, [], net, None, None, dataset, None, None)
        assert len(callback.kernels) == 1

    def test_find_first_convolution(self):
        n = trw.simple_layers.Input([None, 1, 16, 17], feature_name='image')
        convolution_kernels = [(7, 7), (5, 5), (3, 3)]
        n = trw.simple_layers.convs_2d(
            n,
            convolution_kernels=convolution_kernels,
            strides=1,
            pooling_size=2,
            convolution_repeats=1,
            with_flatten=True,
            dropout_probability=0.5,
            channels=[4, 8, 16]
        )
        n = trw.simple_layers.OutputEmbedding(n, 'embedding')
        net = trw.simple_layers.compile_nn(output_nodes=[n])

        batch = {
            'image': torch.ones([10, 1, 16, 17], dtype=torch.float32)
        }

        c = trw.train.find_first_forward_convolution(net, batch)
        kernel_shape = c['matched_module'].weight.shape
        assert kernel_shape == (4, 1, 7, 7)

        c = trw.train.find_first_forward_convolution(net, batch, relative_index=1)
        kernel_shape = c['matched_module'].weight.shape
        assert kernel_shape == (8, 4, 5, 5)

        c = trw.train.find_first_forward_convolution(net, batch, relative_index=2)
        kernel_shape = c['matched_module'].weight.shape
        assert kernel_shape == (16, 8, 3, 3)
        print('DONE')