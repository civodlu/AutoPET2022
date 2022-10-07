import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from torch.nn import init
from trw.layers import BlockConvNormActivation, default_layer_config, NormType
from trw.layers.blocks import BlockUpsampleNnConvNormActivation
from trw.layers.gan import Gan, GanDataPool
from trw.train import OutputEmbedding, OutputLoss, LossMsePacked, apply_spectral_norm, MetricLoss
from trw.train.outputs_trw import OutputClassification
from torch.optim import lr_scheduler


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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        config = default_layer_config(
            conv_kwargs={'padding': 'same', 'bias': True, 'padding_mode': 'reflect'},
            deconv_kwargs={'padding': 'same', 'bias': True, 'padding_mode': 'reflect'},
            norm_type=NormType.InstanceNorm
        )
        channels = [32, 64, 128]
        generator = trw.layers.EncoderDecoderResnet(
            dimensionality=2,
            input_channels=3,
            output_channels=3,
            encoding_channels=channels,
            decoding_channels=list(reversed(channels)),
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

        #l1 = torch.nn.L1Loss(reduction='none')(o, image)
        #l1 = 10.0 * trw.utils.flatten(l1).mean(dim=1)
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
            #('l1', OutputLoss(l1))  # L1 loss
        ])


class SubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        factor = 1
        base_filters = [64, 128, 256, 2]
        filters = [f * factor for f in base_filters]

        config = default_layer_config(
            conv_kwargs={'padding': 'same', 'bias': True, 'padding_mode': 'reflect'},
            deconv_kwargs={'padding': 'same', 'bias': True, 'padding_mode': 'reflect'},
            norm_type=NormType.InstanceNorm
        )
        self.convs = trw.layers.convs_2d(
            6,
            filters,
            convolution_kernels=[4, 4, 4, 1],
            strides=[2, 2, 2, 1],
            pooling_size=None,
            dropout_probability=0.2,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True,
            config=config
        )

    def forward(self, image, segmentation, is_real):
        o = self.convs(torch.cat([image, segmentation], dim=1))
        o_expected = torch.full([o.shape[0]] + list(o.shape[2:]), int(is_real), device=image.device, dtype=torch.long)

        return OutputClassification(
            o, o_expected,
            criterion_fn=LossMsePacked,
        )


class Discriminator(nn.Module):
    def __init__(self, nb_scales=3, discriminator_fn=SubDiscriminator):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for i in range(nb_scales):
            self.discriminators.append(discriminator_fn())

    def forward(self, batch, image, is_real):
        outputs = []
        segmentation = get_segmentation(batch)['segmentation']
        for c in self.discriminators:
            o = c(image, segmentation, is_real)
            outputs.append(o)

            # reduce by half-size each level
            segmentation = nn.functional.avg_pool2d(segmentation, kernel_size=2)
            image = nn.functional.avg_pool2d(image, kernel_size=2)

        outputs_kvp = []
        for o_n, o in enumerate(outputs):
            outputs_kvp.append((f'o_{o_n}', o))

        return collections.OrderedDict(outputs_kvp)





def get_image(batch, source=None):
    return 2 * batch['images'] - 1


def get_segmentation(batch, source=None):
    return {
        'segmentation': 2 * batch['segmentations'] - 1
    }


def optimizer_fn(params, lr):
    optimizer = torch.optim.Adam(lr=lr, betas=(0.5, 0.999), params=params)

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - num_epochs) / float(num_epochs + 1)
        print('LR=', lr_l)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return optimizer, scheduler


def create_model(options):
    latent_size = 0

    discriminator = apply_spectral_norm(Discriminator())
    #discriminator = Discriminator()
    generator = Generator()

    lr_base = 0.0002 #* 10 #* 0.1

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=functools.partial(optimizer_fn, lr=lr_base),  # TTUR
        optimizer_generator_fn=functools.partial(optimizer_fn, lr=lr_base),
        real_image_from_batch_fn=get_image,
        image_pool=GanDataPool(100)
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
        trw.callbacks.CallbackSkipEpoch(20, [
            #trw.callbacks.CallbackReportingLayerStatistics(),
        ], include_epoch_zero=True),
    ]


def pre_training_callbacks():
    return [
        trw.callbacks.CallbackReportingStartServer(),
    ]


num_epochs = 10000
options = trw.train.Options(num_epochs=num_epochs, device='cuda:0')

trainer = trw.train.Trainer(
    callbacks_per_epoch_fn=per_epoch_callbacks,
)

model, result = trw.train.run_trainer_repeat(
    trainer,
    options,
    number_of_training_runs=10,
    inputs_fn=lambda: trw.datasets.create_facades_dataset(
        batch_size=16,
        transforms_train=trw.transforms.TransformCompose([
            trw.transforms.TransformRandomFlip(axis=3),
        ])),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_pix2pix_2_multiscale_discriminator',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')
