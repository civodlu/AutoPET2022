import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from trw.layers import BlockConvNormActivation, default_layer_config, NormType
from trw.layers.gan import Gan
from trw.train import OutputEmbedding, OutputLoss, LossMsePacked, apply_spectral_norm
from trw.train.outputs_trw import OutputClassificationBinary
from torch.optim import lr_scheduler


class Generator(nn.Module):
    def __init__(self, final_activation=torch.tanh):
        super().__init__()
        self.final_activation = final_activation

        self.generator = trw.layers.EncoderDecoderResnet(
            dimensionality=2,
            input_channels=3,
            output_channels=3,
            encoding_channels=[128, 256, 512],
            decoding_channels=[256, 128, 64],
            config=default_layer_config(
                conv_kwargs={'padding': 'same', 'bias': True, 'padding_mode': 'reflect'},
                deconv_kwargs={'padding': 'same', 'bias': True},
                norm_type=NormType.InstanceNorm,
                norm_kwargs={'affine': False},
                activation=nn.ReLU)
        )

    def forward(self, batch, latent):
        segmentation = get_segmentation(batch)['segmentation']
        image = get_image(batch)

        o = self.generator(segmentation)
        if self.final_activation is not None:
            o = self.final_activation(o)

        l1 = 5.0 * torch.nn.L1Loss(reduction='none')(o, image).mean(dim=(1, 2, 3))
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
            ('l1', OutputLoss(l1))  # L1 loss
        ])


def base_discriminator():
    init_block = BlockConvNormActivation(
        config=default_layer_config(
            norm_type=None,
            dimensionality=2,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
        ),
        input_channels=6,
        output_channels=64,
        kernel_size=4,
        stride=2
    )

    convs = trw.layers.convs_2d(
        input_channels=64,
        channels=[128, 256, 1],
        convolution_kernels=4,
        strides=[2, 2, 1],
        convolution_repeats=[3, 2, 1],
        pooling_size=None,
        norm_type=NormType.InstanceNorm,
        last_layer_is_output=True,
        activation=functools.partial(nn.LeakyReLU, negative_slope=0.2)
    )

    return nn.Sequential(init_block, convs)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList()
        for i in range(3):
            self.discriminators.append(base_discriminator())

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, batch, image, is_real):
        segmentation = get_segmentation(batch)['segmentation']

        # introduce the target as one hot encoding input to the netD
        i = torch.cat([image, segmentation], dim=1)

        i_downscaled = i
        outputs = {}
        for d_n, d in enumerate(self.discriminators):
            o = d(i_downscaled)

            o_expected = torch.full(
                [i.shape[0]] + list(o.shape[2:]),
                int(is_real),
                device=image.device, dtype=torch.long
            ).unsqueeze(1)

            outputs[f'classification_{d_n}'] = OutputClassificationBinary(
                o, o_expected,
                criterion_fn=LossMsePacked,  # LSGan loss function
            )

            if d_n + 1 != len(self.discriminators):
                i_downscaled = self.downsample(i_downscaled)

        return outputs


def get_image(batch):
    return 2.0 * batch['images'] - 1


def get_segmentation(batch):
    return {
        'segmentation': 2.0 * batch['segmentations'] - 1
    }


def optimizer_fn(params, lr):
    optimizer = torch.optim.Adam(lr=lr, betas=(0.5, 0.999), params=params)

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - num_epochs // 2) / float(num_epochs // 2 + 1)
        print(f'epoch={epoch}, fLR={lr_l}')
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return optimizer, scheduler


def create_model():
    latent_size = 0

    discriminator = apply_spectral_norm(Discriminator())
    generator = Generator()

    lr_base = 0.0002

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=functools.partial(optimizer_fn, lr=lr_base),  # TTUR
        optimizer_generator_fn=functools.partial(optimizer_fn, lr=lr_base * 0.7),
        real_image_from_batch_fn=get_image,
    )

    return model


def per_epoch_callbacks():
    return [
        trw.callbacks.CallbackSkipEpoch(4, [
            trw.callbacks.CallbackReportingExportSamples(),
        ], include_epoch_zero=True),
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackReportingRecordHistory(),
    ]


def pre_training_callbacks():
    return [
        trw.callbacks.CallbackReportingStartServer(),
    ]


num_epochs = 200
options = trw.train.Options(num_epochs=num_epochs)

trainer = trw.train.TrainerV2(
    callbacks_per_epoch=per_epoch_callbacks(),
)

datasets = trw.datasets.create_facades_dataset(
    batch_size=5,
    nb_workers=4,
    transforms_train=[
        trw.transforms.TransformResize(size=[286, 286]),
        trw.transforms.TransformRandomFlip(axis=3),
        trw.transforms.TransformRandomCropPad(padding=None, shape=[3, 256, 256]),
    ]
)

trainer.fit(
    options,
    datasets=datasets,
    eval_every_X_epoch=20,
    model=create_model(),
    log_path='gan_pix2pix_facades',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')
