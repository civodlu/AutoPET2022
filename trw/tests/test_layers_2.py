import functools

import trw
import torch
import torch.nn as nn
from unittest import TestCase

from trw.layers import default_layer_config, AutoencoderConvolutionalVariationalConditional, EncoderDecoderResnet, \
    NormType
from trw.layers.autoencoder_convolutional_variational import AutoencoderConvolutionalVariational
from trw.layers.blocks import BlockMerge, BlockRes, BlockConvNormActivation, BlockUpsampleNnConvNormActivation
from trw.train import one_hot


class TestLayers2(TestCase):
    def test_layer_denses(self):
        denses = trw.layers.denses([1, 8, 16], dropout_probability=0.1, activation=nn.LeakyReLU, last_layer_is_output=True)
        layers = list(denses)
        assert len(layers) == 6
        assert isinstance(layers[0], trw.layers.Flatten)
        assert isinstance(layers[1], nn.Linear)
        assert isinstance(layers[2], nn.BatchNorm1d)
        assert isinstance(layers[3], nn.LeakyReLU)
        assert isinstance(layers[4], nn.Dropout)
        assert isinstance(layers[5], nn.Linear)

        o = denses(torch.zeros([10, 1]))
        assert o.shape == (10, 16)

    def test_layer_denses_not_last_output(self):
        denses = trw.layers.denses([1, 8, 16], dropout_probability=0.1, activation=nn.LeakyReLU, last_layer_is_output=False)
        layers = list(denses)
        assert len(layers) == 9
        assert isinstance(layers[0], trw.layers.Flatten)
        assert isinstance(layers[1], nn.Linear)
        assert isinstance(layers[2], nn.BatchNorm1d)
        assert isinstance(layers[3], nn.LeakyReLU)
        assert isinstance(layers[4], nn.Dropout)
        assert isinstance(layers[5], nn.Linear)
        assert isinstance(layers[6], nn.BatchNorm1d)
        assert isinstance(layers[7], nn.LeakyReLU)
        assert isinstance(layers[8], nn.Dropout)

        o = denses(torch.zeros([10, 1]))
        assert o.shape == (10, 16)

    def test_conv_3d(self):
        layers = trw.layers.convs_3d(1, [2, 3, 4])
        assert len(layers.layers) == 3

    def test_sub_tensor(self):
        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)

        layer = trw.layers.SubTensor([0, 10, 15], [1, 14, 22])
        o = layer(i)
        assert o.shape == (5, 1, 4, 7)

    def test_crop_or_pad__pad_only(self):
        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)
        i_shaped = trw.layers.crop_or_pad_fun(i, (38, 40))
        assert i_shaped.shape == (5, 1, 38, 40)
        assert (i_shaped[:, :, 3:35, 4:36] == i).all()

    def test_crop_or_pad__crop_only(self):
        i = torch.randn([5, 1, 32, 32], dtype=torch.float32)
        i_shaped = trw.layers.crop_or_pad_fun(i, (16, 20))
        assert i_shaped.shape == (5, 1, 16, 20)
        assert (i[:, :, 8:24, 6:26] == i_shaped).all()

    def test_convs_layers_default(self):
        conf = trw.layers.default_layer_config(2)
        block = trw.layers.BlockConvNormActivation(conf, 1, 8, kernel_size=(5, 5))
        assert len(block.ops) == 3
        assert isinstance(block.ops[0].ops[0], nn.Conv2d)
        assert block.ops[0].ops[0].weight.shape == (8, 1, 5, 5)
        assert isinstance(block.ops[1], nn.BatchNorm2d)
        assert isinstance(block.ops[2], nn.ReLU)

    def test_convs_additional_args(self):
        conf = trw.layers.default_layer_config(
            2,
            norm_type=trw.layers.NormType.BatchNorm,
            conv_kwargs={'bias': False},
            norm_kwargs={'eps': 0.001},
            activation=nn.ReLU,
            activation_kwargs={'inplace': True}
        )

        block = trw.layers.BlockConvNormActivation(conf, 1, 8, kernel_size=5, stride=2)
        assert len(block.ops) == 3
        assert isinstance(block.ops[0].ops[0], nn.Conv2d)
        assert block.ops[0].ops[0].weight.shape == (8, 1, 5, 5)
        assert block.ops[0].ops[0].bias is None
        assert block.ops[0].ops[0].stride == (2, 2)
        assert isinstance(block.ops[1], nn.BatchNorm2d)
        assert block.ops[1].eps == 0.001
        assert isinstance(block.ops[2], nn.ReLU)
        assert block.ops[2].inplace

    def test_conv_multiple_kernel_sizes(self):
        net = trw.layers.ConvsBase(
            2, 1, channels=[4, 8, 10],
            convolution_kernels=[5, 7, 9],
            strides=2,
            pooling_size=None,
            last_layer_is_output=True
        )

        assert len(net.layers) == 3

        block = net.layers[0]
        assert len(block) == 1  # there is no pooling or dropout
        ops = list(block[0].ops)
        assert isinstance(block[0], trw.layers.BlockConvNormActivation)
        assert len(ops) == 3
        assert isinstance(ops[0].ops[0], nn.Conv2d)
        assert ops[0].ops[0].padding == (2, 2)
        assert isinstance(ops[1], nn.BatchNorm2d)
        assert isinstance(ops[2], nn.ReLU)

        block = net.layers[1]
        assert len(block) == 1  # there is no pooling or dropout
        ops = list(block[0].ops)
        assert isinstance(block[0], trw.layers.BlockConvNormActivation)
        assert len(ops) == 3
        assert isinstance(ops[0].ops[0], nn.Conv2d)
        assert ops[0].ops[0].padding == (3, 3)
        assert isinstance(ops[1], nn.BatchNorm2d)
        assert isinstance(ops[2], nn.ReLU)

        block = net.layers[2]
        assert len(block) == 1  # there is no pooling or dropout
        ops = list(block[0].ops)
        assert isinstance(block[0], trw.layers.BlockConvNormActivation)
        assert len(ops) == 1
        assert isinstance(ops[0].ops[0], nn.Conv2d)
        assert ops[0].ops[0].padding == (4, 4)

        o = net(torch.zeros([5, 1, 64, 64]))
        assert o.shape == (5, 10, 64 // 2 ** 3, 64 // 2 ** 3)

    def test_conv2(self):
        net = trw.layers.ConvsBase(
            3, 2, channels=[4, 8, 10],
            convolution_kernels=3,
            strides=2,
            pooling_size=2,
            norm_type=trw.layers.NormType.InstanceNorm,
            convolution_repeats=2,
            activation=nn.LeakyReLU,
            with_flatten=True,
            config=default_layer_config(dimensionality=None),
            last_layer_is_output=True)

        #
        # First Block
        #
        assert len(net.layers) == 3
        children_0 = list(net.layers[0].children())
        assert len(children_0) == 3

        children_00 = list(net.layers[0][0].ops)
        assert len(children_00) == 3
        assert isinstance(children_00[0].ops[0], nn.Conv3d)
        assert children_00[0].ops[0].weight.shape[1] == 2
        assert children_00[0].ops[0].stride == (1, 1, 1)
        assert isinstance(children_00[1], nn.InstanceNorm3d)
        assert isinstance(children_00[2], nn.LeakyReLU)

        children_01 = list(net.layers[0][1].ops)
        assert len(children_01) == 3
        assert isinstance(children_01[0].ops[0], nn.Conv3d)
        assert children_01[0].ops[0].weight.shape[1] == 4
        assert children_01[0].ops[0].stride == (2, 2, 2)
        assert isinstance(children_01[1], nn.InstanceNorm3d)
        assert isinstance(children_01[2], nn.LeakyReLU)

        children_02 = net.layers[0][2].op
        assert isinstance(children_02, nn.MaxPool3d)

        #
        # Second block
        #
        children_1 = list(net.layers[1].children())
        assert len(children_1) == 3

        children_10 = list(net.layers[1][0].ops)
        assert len(children_10) == 3
        assert isinstance(children_10[0].ops[0], nn.Conv3d)
        assert children_10[0].ops[0].weight.shape[1] == 4
        assert children_10[0].ops[0].stride == (1, 1, 1)
        assert isinstance(children_10[1], nn.InstanceNorm3d)
        assert isinstance(children_10[2], nn.LeakyReLU)

        children_11 = list(net.layers[1][1].ops)
        assert len(children_11) == 3
        assert isinstance(children_11[0].ops[0], nn.Conv3d)
        assert children_11[0].ops[0].weight.shape[1] == 8
        assert children_11[0].ops[0].stride == (2, 2, 2)
        assert isinstance(children_11[1], nn.InstanceNorm3d)
        assert isinstance(children_11[2], nn.LeakyReLU)

        children_12 = net.layers[1][2].op
        assert isinstance(children_12, nn.MaxPool3d)

        #
        # Third block
        #
        children_2 = list(net.layers[2].children())
        assert len(children_2) == 3

        children_20 = list(net.layers[2][0].ops)
        assert len(children_20) == 3
        assert isinstance(children_20[0].ops[0], nn.Conv3d)
        assert children_20[0].ops[0].stride == (1, 1, 1)
        assert isinstance(children_20[1], nn.InstanceNorm3d)
        assert isinstance(children_20[2], nn.LeakyReLU)

        children_21 = list(net.layers[2][1].ops)
        assert len(children_21) == 1
        assert isinstance(children_21[0].ops[0], nn.Conv3d)
        assert children_11[0].ops[0].stride == (2, 2, 2)

        children_22 = net.layers[2][2].op
        assert isinstance(children_22, nn.MaxPool3d)

    def test_deconv_block(self):
        conf = trw.layers.default_layer_config(2)
        block = trw.layers.BlockDeconvNormActivation(conf, 1, 8, kernel_size=(5, 5))
        assert len(block.ops) == 3
        assert isinstance(block.ops[0], nn.ConvTranspose2d)
        assert block.ops[0].weight.shape == (1, 8, 5, 5)
        assert isinstance(block.ops[1], nn.BatchNorm2d)
        assert isinstance(block.ops[2], nn.ReLU)

    def test_conv_block_4_isotropic_kernel(self):
        """
        with padding == 4, we need additional asymmetric padding
        """
        conf = trw.layers.default_layer_config(2)
        block = trw.layers.BlockConvNormActivation(conf, 1, 8, kernel_size=4)
        assert len(block.ops) == 3
        assert len(block.ops[0].ops) == 2
        assert isinstance(block.ops[0].ops[0], nn.ConstantPad2d)
        assert block.ops[0].ops[0].padding == (2, 1, 2, 1)

        assert isinstance(block.ops[0].ops[1], nn.Conv2d)
        assert block.ops[0].ops[1].weight.shape == (8, 1, 4, 4)
        assert isinstance(block.ops[1], nn.BatchNorm2d)
        assert isinstance(block.ops[2], nn.ReLU)

        i = torch.zeros([4, 1, 6, 6])
        o = block(i)
        assert o.shape == (4, 8, 6, 6)

    def test_conv_block_4_anisotropic_kernel(self):
        """
        with padding == 4, we need additional asymmetric padding
        """
        conf = trw.layers.default_layer_config(2)
        block = trw.layers.BlockConvNormActivation(conf, 1, 8, kernel_size=(4, 6))
        assert len(block.ops) == 3
        assert len(block.ops[0].ops) == 2
        assert isinstance(block.ops[0].ops[0], nn.ConstantPad2d)
        assert block.ops[0].ops[0].padding == (3, 2, 2, 1)

        assert isinstance(block.ops[0].ops[1], nn.Conv2d)
        assert block.ops[0].ops[1].weight.shape == (8, 1, 4, 6)
        assert isinstance(block.ops[1], nn.BatchNorm2d)
        assert isinstance(block.ops[2], nn.ReLU)

        i = torch.zeros([2, 1, 10, 10])
        o = block(i)
        assert o.shape == (2, 8, 10, 10)

    def test_convs_transpose(self):
        model = trw.layers.ConvsTransposeBase(2, input_channels=1, channels=[4, 8, 16], last_layer_is_output=True, dropout_probability=0.1)
        i = torch.zeros([5, 1, 3, 3])
        o = model(i)
        assert o.shape == (5, 16, 3 * 2 ** 3, 3 * 2 ** 3)

        assert len(model.layers) == 3

        layer = model.layers[0]
        assert len(layer) == 2
        layer_0 = layer[0]
        assert isinstance(layer_0, trw.layers.BlockDeconvNormActivation)
        lasyer_0_ops = list(layer_0.ops)
        assert len(lasyer_0_ops) == 3
        assert isinstance(lasyer_0_ops[0], nn.ConvTranspose2d)
        assert isinstance(lasyer_0_ops[1], nn.BatchNorm2d)
        assert isinstance(lasyer_0_ops[2], nn.ReLU)
        layer_1 = layer[1]
        assert isinstance(layer_1, nn.Dropout)

        layer = model.layers[1]
        assert len(layer) == 2
        layer_0 = layer[0]
        assert isinstance(layer_0, trw.layers.BlockDeconvNormActivation)
        lasyer_0_ops = list(layer_0.ops)
        assert len(lasyer_0_ops) == 3
        assert isinstance(lasyer_0_ops[0], nn.ConvTranspose2d)
        assert isinstance(lasyer_0_ops[1], nn.BatchNorm2d)
        assert isinstance(lasyer_0_ops[2], nn.ReLU)
        layer_1 = layer[1]
        assert isinstance(layer_1, nn.Dropout)

        layer = model.layers[2]
        assert len(layer) == 1
        layer_0 = layer[0]
        assert isinstance(layer_0, trw.layers.BlockDeconvNormActivation)
        lasyer_0_ops = list(layer_0.ops)
        assert len(lasyer_0_ops) == 1
        assert isinstance(lasyer_0_ops[0], nn.ConvTranspose2d)

    def test_convs_fcnn(self):
        convs = trw.layers.ConvsBase(input_channels=1, dimensionality=2, channels=[16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            dimensionality=2,
            base_model=convs,
            input_channels=64,
            deconv_filters=[32, 16, 8],
            convolution_kernels=3,
            strides=2,
            nb_classes=2
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_convs_fcnn_concatenate_mode(self):
        convs = trw.layers.ConvsBase(dimensionality=2, input_channels=1, channels=[16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            dimensionality=2,
            base_model=convs,
            input_channels=64,
            deconv_filters=[32, 16, 8],
            conv_filters=[16, 32, 64],
            convolution_kernels=3,
            strides=2,
            nb_classes=2,
            concat_mode='concatenate'
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_convs_fcnn_different_kernels(self):
        convs = trw.layers.ConvsBase(dimensionality=2, input_channels=1, channels=[16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            dimensionality=2,
            base_model=convs,
            input_channels=64,
            deconv_filters=[32, 16, 8],
            convolution_kernels=[3, 5, 3],
            strides=[2] * 3,
            nb_classes=2
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_convs_fcnn_kernel7(self):
        convs = trw.layers.ConvsBase(dimensionality=2, input_channels=1, channels=[16, 32, 64])
        fcnn = trw.layers.FullyConvolutional(
            dimensionality=2,
            base_model=convs,
            input_channels=64,
            deconv_filters=[32, 16, 8],
            convolution_kernels=7,
            strides=[2] * 3,
            nb_classes=2
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        o = fcnn(i)
        assert o.shape == (5, 2, 32, 32)

    def test_autoencoder_conv_padding(self):
        model = trw.layers.AutoencoderConvolutional(
            2, 1,
            encoder_channels=[4, 8, 16],
            decoder_channels=[8, 4, 1],
            last_layer_is_output=True
        )

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        intermediates = model.forward_with_intermediate(i)
        assert len(intermediates) == 2
        encoding, reconstruction = intermediates

        assert encoding.shape == (5, 16, 4, 4)
        assert reconstruction.shape == i.shape

        last_op = model.decoder.layers[-1][-1].ops[-1]
        assert isinstance(last_op, nn.ConvTranspose2d)

    def test_autoencoder_conv_cropping(self):
        model = trw.layers.AutoencoderConvolutional(2, 1, [4, 8, 16], [16, 8, 4, 1], last_layer_is_output=True)

        i = torch.zeros([5, 1, 32, 32], dtype=torch.float32)
        intermediates = model.forward_with_intermediate(i)
        assert len(intermediates) == 2
        encoding, reconstruction = intermediates

        assert encoding.shape == (5, 16, 4, 4)
        assert reconstruction.shape == i.shape

        last_op = model.decoder.layers[-1][-1].ops[-1]
        assert isinstance(last_op, nn.ConvTranspose2d)

    def test_autoencoder_conv_var(self):
        z_size = 20

        encoder = trw.layers.ConvsBase(
            dimensionality=2,
            input_channels=1,
            channels=[8, 16, 32],
            convolution_kernels=3,
            norm_type=None,
        )

        decoder = trw.layers.ConvsTransposeBase(
            dimensionality=2,
            input_channels=z_size,
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 2],
            convolution_kernels=3,
            last_layer_is_output=True,
            squash_function=torch.sigmoid,
        )

        x = torch.randn([10, 1, 28, 28], dtype=torch.float32)
        model = AutoencoderConvolutionalVariational([1, 1, 28, 28], encoder, decoder, z_size)
        recon, mu, logvar = model(x)

        assert recon.shape == (10, 1, 28, 28)
        assert mu.shape == (10, 20)
        assert mu.shape == logvar.shape

        loss_bce = AutoencoderConvolutionalVariational.loss_function(recon, x, mu, logvar, recon_loss_name='BCE')
        assert loss_bce.shape == (10,)
        loss_mse = AutoencoderConvolutionalVariational.loss_function(recon, x, mu, logvar, recon_loss_name='MSE')
        assert loss_mse.shape == (10,)

    def test_autoencoder_conv_var_conditional(self):
        z_size = 20
        y_size = 5

        encoder = trw.layers.ConvsBase(
            dimensionality=2,
            input_channels=1,
            channels=[8, 16, 32],
            convolution_kernels=3,
            norm_type=None
        )

        decoder = trw.layers.ConvsTransposeBase(
            dimensionality=2,
            input_channels=z_size + y_size,
            channels=[32, 16, 8, 1],
            strides=[2, 2, 2, 2],
            convolution_kernels=3,
            last_layer_is_output=True,
            squash_function=torch.sigmoid,
        )

        y = one_hot(torch.tensor([0] * 10, dtype=torch.long), y_size)
        x = torch.randn([10, 1, 28, 28], dtype=torch.float32)
        model = AutoencoderConvolutionalVariationalConditional([1, 1, 28, 28], encoder, decoder, z_size, y_size=y_size)
        recon, mu, logvar = model(x, y)

        assert recon.shape == (10, 1, 28, 28)
        assert mu.shape == (10, z_size)
        assert mu.shape == logvar.shape

    def test_layer_res(self):
        config = default_layer_config(dimensionality=2)
        b = BlockRes(config, 8, kernel_size=7, padding='same', padding_mode='reflect')

        i = torch.zeros([2, 8, 16, 16])
        o = b(i)
        assert o.shape == (2, 8, 16, 16)

        version = torch.__version__[:3]

        ops_b = list(b.block_1.ops)
        assert len(ops_b) == 3
        assert isinstance(ops_b[0].ops[0], torch.nn.Conv2d)
        if version != '1.0':
            assert ops_b[0].ops[0].padding_mode == 'reflect'
        assert isinstance(ops_b[1], torch.nn.BatchNorm2d)
        assert isinstance(ops_b[2], torch.nn.ReLU)

        ops_b = list(b.block_2.ops)
        assert len(ops_b) == 2
        assert isinstance(ops_b[0].ops[0], torch.nn.Conv2d)
        if version != '1.0':
            assert ops_b[0].ops[0].padding_mode == 'reflect'
        assert isinstance(ops_b[1], torch.nn.BatchNorm2d)

    def test_encoder_decoder_res(self):
        config = default_layer_config(
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same', 'bias': False}
        )
        I_O = functools.partial(BlockConvNormActivation, kernel_size=7)
        model = EncoderDecoderResnet(
            2, 3, 2,
            encoding_channels=[16, 32, 64],
            decoding_channels=[64, 32, 16],
            convolution_kernel=5,
            init_block=I_O,
            out_block=I_O,
            config=config,
            activation=nn.LeakyReLU
        )

        t = torch.zeros([5, 3, 32, 64])
        o = model(t)
        assert o.shape == (5, 2, 32, 64)

        assert len(model.initial.ops) == 3
        assert model.initial.ops[0].ops[0].kernel_size == (7, 7)
        assert model.initial.ops[0].ops[0].bias is None
        assert isinstance(model.initial.ops[2], nn.LeakyReLU)

        assert len(model.encoders) == 3

        ops = model.encoders[0].ops
        assert len(ops) == 3
        assert ops[0].ops[0].kernel_size == (5, 5)
        assert ops[0].ops[0].bias is None
        assert isinstance(ops[2], nn.LeakyReLU)

        ops = model.encoders[1].ops
        assert len(ops) == 3
        assert ops[0].ops[0].kernel_size == (5, 5)
        assert ops[0].ops[0].bias is None
        assert isinstance(ops[2], nn.LeakyReLU)

        ops = model.encoders[2].ops
        assert len(ops) == 3
        assert ops[0].ops[0].kernel_size == (5, 5)
        assert ops[0].ops[0].bias is None
        assert isinstance(ops[2], nn.LeakyReLU)

        assert len(model.decoders) == 3

        ops = model.decoders[0].ops
        assert len(ops) == 3
        assert ops[0].kernel_size == (5, 5)
        assert ops[0].bias is None
        assert isinstance(ops[2], nn.LeakyReLU)

        ops = model.decoders[1].ops
        assert len(ops) == 3
        assert ops[0].kernel_size == (5, 5)
        assert ops[0].bias is None
        assert isinstance(ops[2], nn.LeakyReLU)

        ops = model.decoders[2].ops
        assert len(ops) == 3
        assert ops[0].kernel_size == (5, 5)
        assert ops[0].bias is None
        assert isinstance(ops[2], nn.LeakyReLU)

        ops = model.out.ops
        assert len(ops) == 1
        assert len(ops[0].ops) == 1
        assert ops[0].ops[0].kernel_size == (7, 7)
        assert ops[0].ops[0].bias is None

    def test_encoder_decoder_res_k4(self):
        config = default_layer_config(
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same', 'bias': False}
        )
        I_O = functools.partial(BlockConvNormActivation, kernel_size=7)
        model = EncoderDecoderResnet(
            2, 3, 2,
            encoding_channels=[16, 32, 64],
            decoding_channels=[64, 32, 16],
            convolution_kernel=4,
            init_block=I_O,
            out_block=I_O,
            config=config,
            decoding_block=BlockUpsampleNnConvNormActivation,
            activation=nn.LeakyReLU
        )

        t = torch.zeros([5, 3, 32, 64])
        o = model(t)
        assert o.shape == (5, 2, 32, 64)

    def test_upsample_nn(self):
        config = default_layer_config(dimensionality=3, norm_type=NormType.InstanceNorm)
        block = BlockUpsampleNnConvNormActivation(config, 2, 4, kernel_size=3, stride=(1, 2, 3))

        i = torch.zeros([10, 2, 4, 5, 6], dtype=torch.float32)
        o = block(i)
        assert o.shape == (10, 4, 4 * 1, 5 * 2, 6 * 3)

        ops = list(block.ops)
        assert len(ops) == 4
        assert isinstance(ops[0], nn.Upsample)
        assert isinstance(ops[1], nn.Conv3d)
        assert isinstance(ops[2], nn.InstanceNorm3d)
        assert isinstance(ops[3], nn.ReLU)

    def test_deep_supervision(self):
        backbone = trw.layers.UNetBase(dim=2, input_channels=3, channels=[2, 4, 8], output_channels=2)
        deep_supervision = trw.layers.DeepSupervision(
            backbone,
            [3, 8, 16],
        )

        i = torch.zeros([1, 3, 8, 16], dtype=torch.float32)
        t = torch.zeros([1, 1, 8, 16], dtype=torch.long)
        outputs = deep_supervision(i, t)
        assert len(outputs) == 2
        assert isinstance(outputs[0], trw.train.OutputSegmentation)
        assert outputs[0].output.shape == (1, 2, 8, 16)
        assert isinstance(outputs[1], trw.train.OutputSegmentation)
        assert outputs[1].output.shape == (1, 2, 8, 16)

    def test_backbone_upsampler(self):
        encoder = trw.layers.convs_2d(1, [4, 8, 16])
        encoder_decoder = trw.layers.BackboneDecoder(
            output_channels=6,
            backbone=encoder,
            decoding_channels=(5, 4, 3),
            backbone_transverse_connections=(0, 1, 2),
            backbone_input_shape=(1, 1, 16, 24))

        # `backbone_input_shape` is just used to calculate channels,
        # we must be able to use for different sizes
        r = encoder_decoder.forward(torch.zeros((2, 1, 32, 48), dtype=torch.float32))

        print('DONE')
        assert r.shape == (2, 6, 32, 48)

    def test_backbone_upsampler_latent(self):
        encoder = trw.layers.convs_2d(1, [4, 8, 16])
        encoder_decoder = trw.layers.BackboneDecoder(
            output_channels=6,
            backbone=encoder,
            decoding_channels=(5, 4, 3),
            backbone_transverse_connections=(0, 1, 2),
            latent_channels=2,
            backbone_input_shape=(1, 1, 16, 24))

        # use a latent
        latent = torch.zeros([2, 2, 1, 1], dtype=torch.float32)
        r = encoder_decoder.forward(torch.zeros((2, 1, 32, 48), dtype=torch.float32), latent=latent)
        assert r.shape == (2, 6, 32, 48)

    def test_skip_up_deconv(self):
        config = default_layer_config(dimensionality=2)
        l = trw.layers.BlockUpDeconvSkipConv(
            config,
            skip_channels=3,
            input_channels=1,
            output_channels=3,
            nb_repeats=4,
            merge_layer_fn=functools.partial(BlockMerge, mode='sum'),
            kernel_size=3,
            padding=1,
            output_padding=1,
            stride=2
        )

        skip = torch.zeros([4, 3, 32, 32], dtype=torch.float32)
        down = torch.zeros([4, 1, 16, 16], dtype=torch.float32)
        up = l(skip, down)
        assert up.shape == (4, 3, 32, 32)
        assert len(l.ops_conv) == 4
