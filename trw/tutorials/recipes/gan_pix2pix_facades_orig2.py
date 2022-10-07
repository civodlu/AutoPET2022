import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools
from torch.nn import init
from trw.layers import BlockConvNormActivation, default_layer_config
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
            conv_kwargs={'padding': 'same', 'bias': False},
            deconv_kwargs={'padding': 'same', 'bias': False}
        )
        generator = trw.layers.EncoderDecoderResnet(
            dimensionality=2,
            input_channels=3,
            output_channels=3,
            encoding_channels=[64, 128, 256],
            decoding_channels=[256, 128, 64],
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
            padding=1,
            dropout_probability=0.2,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True,
        )

    def forward(self, batch, image, is_real):
        segmentation = get_segmentation(batch)['segmentation']

        # introduce the target as one hot encoding input to the discriminator
        o = self.convs(torch.cat([image, segmentation], dim=1))
        #o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)
        #o_expected = torch.zeros(len(image), device=image.device, dtype=torch.float32)
        #if is_real:
        #    # one sided label smoothing  # TODO integrade 1-sided smoothing!
        #    o_expected.uniform_(0.8, 1.2)

        o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long)
        o_expected = o_expected.unsqueeze(1).unsqueeze(1)
        o_expected = o_expected.repeat([1, o.shape[2], o.shape[3]])
        return {
            'classification': OutputClassification(
                o, o_expected,
                criterion_fn=LossMsePacked,  # LSGan loss function
            )
        }


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
            trw.callbacks.CallbackReportingLayerStatistics(),
        ], include_epoch_zero=True),
    ]


def pre_training_callbacks():
    return [
        trw.callbacks.CallbackReportingStartServer(),
    ]


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
        batch_size=1,
        transforms_train=trw.transforms.TransformCompose([
            trw.transforms.TransformRandomFlip(axis=3),
        ])),
    eval_every_X_epoch=20,
    model_fn=create_model,
    run_prefix='facade_pix2pix_3',
    optimizers_fn=None  # the module has its own optimizers
)

print('DONE')
