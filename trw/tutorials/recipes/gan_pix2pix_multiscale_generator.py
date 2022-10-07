import collections
import copy
from typing import List, Optional, Sequence
from typing_extensions import Literal

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from torch.nn import init
from trw.basic_typing import ShapeCX, Stride, ConvKernels, Activation, Paddings
from trw.layers import BlockConvNormActivation, default_layer_config, NormType, ModuleWithIntermediate, LayerConfig
from trw.layers.blocks import BlockUpsampleNnConvNormActivation, ConvTransposeBlockType
from trw.layers.gan import Gan, GanDataPool
from trw.train import OutputEmbedding, OutputLoss, LossMsePacked, apply_spectral_norm, MetricLoss
from trw.train.outputs_trw import OutputClassification
from torch.optim import lr_scheduler
from trw.transforms import resize


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class UpsamplingGenerator(nn.Module):
    """
    GauGan like generator

    Instead of down-sampling the conditional images and upsampling with skip connection,
    start from a down-sampled version of the conditional image and up-sample it.
    """
    def __init__(
            self,
            dimensionality: int,
            input_channels: int,
            channels: Sequence[int],
            conditional_channels: int,
            blocks_with_supervision: Sequence[int],
            *,
            kernel_size: ConvKernels = 3,
            stride: Optional[Stride] = 2,
            resize_mode: Literal['nearest', 'linear'] = 'nearest',
            upsampling_block: ConvTransposeBlockType = BlockUpsampleNnConvNormActivation,
            last_layer_is_output: bool = True,
            config: LayerConfig = default_layer_config()):

        super().__init__()
        self.dimensionality = dimensionality
        self.channels = channels
        self.input_channels = input_channels
        self.resize_mode = resize_mode
        self.conditional_channels = conditional_channels
        assert len(blocks_with_supervision) >= 1

        config = copy.deepcopy(config)
        config.set_dim(dimensionality)

        if stride is not None:
            if isinstance(stride, list):
                assert len(stride) == len(channels)
            else:
                stride = [stride] * len(channels)

        if kernel_size is not None:
            config.conv_kwargs['kernel_size'] = kernel_size
            config.deconv_kwargs['kernel_size'] = kernel_size

        self.blocks = nn.ModuleList()
        self.blocks_with_supervision = sorted(blocks_with_supervision)
        supervision_index = 0
        i = input_channels
        for block_index, o in enumerate(channels):
            if supervision_index < len(self.blocks_with_supervision) and \
                    block_index == self.blocks_with_supervision[supervision_index]:
                # we will receive additional supervision here. Increase the
                # channels by the number of input channels. Supervision MUST be a
                # down-scaled version of the full scale conditional image
                i = i + self.conditional_channels
                supervision_index += 1

            if stride is not None:
                config.deconv_kwargs['stride'] = stride[block_index]

            if last_layer_is_output and block_index + 1 == len(channels):
                # last block: remove activation and normalization layers
                config.activation = None
                config.norm = None

            block = upsampling_block(
                config=config,
                input_channels=i,
                output_channels=o
            )
            i = o
            self.blocks.append(block)

    def forward(
            self,
            x: torch.Tensor,
            conditional: torch.Tensor) -> torch.Tensor:
        outputs = self.forward_with_intermediate(x, conditional=conditional)
        return outputs[-1]

    def forward_with_intermediate(
            self,
            x: torch.Tensor,
            conditional: torch.Tensor) -> List[torch.Tensor]:

        if x is not None:
            assert x.shape[1] == self.input_channels
        else:
            assert self.blocks_with_supervision[0] == 0, 'not embedding! Must have supervision as input for block 0!'
        assert conditional.shape[1] == self.conditional_channels

        outputs = []
        supervision_index = 0
        for block_index, b in enumerate(self.blocks):
            if supervision_index < len(self.blocks_with_supervision) and \
                    block_index == self.blocks_with_supervision[supervision_index]:
                # reshape the conditional input to the input shape and concatenate to current input
                with torch.no_grad():
                    # we don't want gradient propagation through the resizing
                    reshaped_conditional = resize(conditional, x.shape[2:], mode=self.resize_mode)

                x = torch.cat([x, reshaped_conditional], dim=1)
                supervision_index += 1

            x = b(x)
            outputs.append(x)
        return outputs


class Generator_GOOD(nn.Module):
    def __init__(self):
        super().__init__()

        config = default_layer_config(
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same', 'bias': False}
        )
        generator = trw.layers.EncoderDecoderResnet(
            dimensionality=2,
            input_channels=3,
            output_channels=3,
            convolution_kernel=3,
            encoding_channels=[64, 128, 256],
            decoding_channels=[256, 128, 64],
            decoding_block=BlockUpsampleNnConvNormActivation,
            init_block=functools.partial(BlockConvNormActivation, kernel_size=7),
            out_block=functools.partial(BlockConvNormActivation, kernel_size=7),
            config=config
        )
        self.generator = generator

    def forward(self, batch, latent):
        segmentation = get_segmentation(batch)['segmentation']
        image = get_image(batch)

        o = self.generator(segmentation)
        o = torch.tanh(o)  # force -1..1 range

        l1 = torch.nn.L1Loss(reduction='none')(o, image)
        l1 = 10.0 * trw.utils.flatten(l1).mean(dim=1)
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
            ('l1', OutputLoss(l1))  # L1 loss
        ])


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        config = default_layer_config(dimensionality=2, norm_type=NormType.InstanceNorm)
        #config = default_layer_config(norm_type=NormType.BatchNorm)

        """
        channels = [32, 64, 128, 256]
        generator = UpsamplingGenerator(
            dimensionality=2,
            input_channels=0,
            channels=channels + [3],
            stride=[2] * len(channels) + [1],
            conditional_channels=3,
            config=config,
            last_layer_is_output=True,
            blocks_with_supervision=[0, 3]
        )
        """
        output_channels = 3
        channels = [64, 128, 256]
        intermediates = [128, 64, 32]
        self.intermediates_outputs = nn.ModuleList()
        for i in intermediates:
            config_intermediate = copy.deepcopy(config)
            config_intermediate.activation = None
            config_intermediate.norm = None
            self.intermediates_outputs.append(trw.layers.BlockConvNormActivation(
                config=config_intermediate,
                input_channels=i,
                output_channels=output_channels,
                kernel_size=1
            ))

        """
        generator = trw.layers.EncoderDecoderResnet(
            dimensionality=2,
            input_channels=3,
            output_channels=3,
            convolution_kernel=3,
            encoding_channels=channels,
            decoding_channels=list(reversed(channels)),
            decoding_block=BlockUpsampleNnConvNormActivation,
            init_block=functools.partial(BlockConvNormActivation, kernel_size=7),
            out_block=functools.partial(BlockConvNormActivation, kernel_size=7),
            config=config
        )
        """

        generator = trw.layers.UNetBase(
            dim=2,
            input_channels=3,
            channels=channels,
            output_channels=3,
            config=config
        )

        self.generator = generator
        self.scale = 2 ** len(channels)

    def forward(self, batch, latent):
        segmentation = get_segmentation(batch)['segmentation']
        image = get_image(batch)

        # create a 0-image: use the supervision as input
        """
        shape = [segmentation.shape[0]] + [0, segmentation.shape[2] // self.scale, segmentation.shape[3] // self.scale]
        empty_image = torch.zeros(shape, dtype=segmentation.dtype, device=segmentation.device)
        o = self.generator(
            empty_image,
            conditional=segmentation)
        o = torch.tanh(o)  # force -1..1 range
        """

        #o = self.generator(segmentation)
        #o = torch.tanh(o)  # force -1..1 range

        os = self.generator.forward_with_intermediate(segmentation)
        os_outputs = []
        for i, o in enumerate(os):
            if i + 1 < len(os):
                o = self.intermediates_outputs[i](o)
            o = torch.tanh(o)  # force -1..1 range
            os_outputs.append(o)

        o = os_outputs[-1]


        l1 = torch.nn.L1Loss(reduction='none')(o, image)
        l1 = 10.0 * trw.utils.flatten(l1).mean(dim=1)
        return os_outputs, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
            ('l1', OutputLoss(l1))  # L1 loss
        ])


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        factor = 1
        base_filters = [64, 128, 256, 512, 512, 2]
        filters = [f * factor for f in base_filters]
        self.convs = trw.layers.convs_2d(
            6,
            filters,
            convolution_kernels=[4, 4, 4, 4, 4, 1],
            strides=[2, 2, 2, 2, 1, 1],
            pooling_size=None,
            dropout_probability=0.2,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.1),
            last_layer_is_output=True,
            config=default_layer_config(dimensionality=None, norm_type=NormType.InstanceNorm)
        )

        self.shapes = [64, 128, 256, 256]
        self.networks = nn.ModuleList()
        for s in self.shapes:
            self.networks.append(
                trw.layers.convs_2d(
                    6,
                    filters,
                    convolution_kernels=[4, 4, 4, 4, 4, 1],
                    strides=[2, 2, 2, 2, 1, 1],
                    pooling_size=None,
                    dropout_probability=0.2,
                    activation=functools.partial(nn.LeakyReLU, negative_slope=0.1),
                    last_layer_is_output=True,
                    config=default_layer_config(dimensionality=None, norm_type=NormType.InstanceNorm)
                )
            )

    def forward(self, batch, image, is_real):
        segmentation = get_segmentation(batch)['segmentation']

        if isinstance(image, list):
            # generator: check we have the expected sizes
            images = image
            assert len(images) == len(self.shapes)
            for index, i in enumerate(images):
                assert i.shape[2] == self.shapes[index]
        else:
            images = []
            for s in self.shapes:
                images.append(resize(image, (s, s)))

        outputs = []
        for i, image in enumerate(images):
            s = resize(segmentation, image.shape[2:])
            o = self.networks[i](torch.cat([image, s], dim=1))

            o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)
            o_expected = o_expected.unsqueeze(1).unsqueeze(1)
            o_expected = o_expected.repeat([1, o.shape[2], o.shape[3]])
            output = OutputClassification(o, o_expected, criterion_fn=LossMsePacked)
            outputs.append((f'c_{i}', output))

        del outputs[-1]  # before the last is NOT useful
        return collections.OrderedDict(outputs)


def get_image(batch, source=None):
    return 2 * batch['images'] - 1


def get_segmentation(batch, source=None):
    return {
        'segmentation': 2 * batch['segmentations'] - 1
    }


def optimizer_fn(params, lr):
    optimizer = torch.optim.Adam(lr=lr, betas=(0.5, 0.999), params=params)

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - 200) / float(200 + 1)
        print('LR=', lr_l)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return optimizer, scheduler


def create_model(options):
    latent_size = 0

    #discriminator = apply_spectral_norm(Discriminator())
    discriminator = Discriminator()
    generator = Generator()

    lr_base = 0.0002
    #lr_base = 0.002

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=functools.partial(optimizer_fn, lr=lr_base),  # TTUR
        optimizer_generator_fn=functools.partial(optimizer_fn, lr=lr_base),
        real_image_from_batch_fn=get_image,
        #image_pool=GanDataPool(100)
    )

    model.apply(init_weights)
    return model


def per_epoch_callbacks():
    return [
        trw.callbacks.CallbackSkipEpoch(4, [
            trw.callbacks.CallbackReportingExportSamples(),
        ], include_epoch_zero=True),
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackReportingRecordHistory(),
        #trw.callbacks.CallbackSkipEpoch(20, [
        #    trw.callbacks.CallbackReportingLayerStatistics(),
        #], include_epoch_zero=True),
    ]


def pre_training_callbacks():
    return [
        trw.callbacks.CallbackReportingStartServer(),
    ]


"""
g = UpsamplingGenerator(
    dimensionality=2,
    input_channels=3,
    channels=(16, 32, 64, 128),
    conditional_channels=1,
    blocks_with_supervision=[0, 2, 3])

x = torch.zeros([5, 3, 8, 8], dtype=torch.float32)
c = torch.zeros([5, 1, 8 * 2 ** 4 , 8 * 2 ** 4], dtype=torch.float32)
outputs = g(x, conditional=c)
"""

num_epochs = 300
options = trw.train.Options(num_epochs=num_epochs, device='cuda:0')

trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=10,
    inputs_fn=lambda: trw.datasets.create_facades_dataset(
        batch_size=32,
        transforms_train=trw.transforms.TransformCompose([
            trw.transforms.TransformRandomFlip(axis=3),
        ])),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_pix2pix2_tmp',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')
