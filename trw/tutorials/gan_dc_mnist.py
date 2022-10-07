import collections

import trw.train
import trw.datasets
import torch
import torch.nn as nn
import functools

from trw.layers.gan import Gan
from trw.train import LossMsePacked, OutputEmbedding
from trw.train.outputs_trw import OutputClassification
from trw.utils import global_average_pooling_2d


def per_epoch_callbacks():
    return [
        trw.callbacks.CallbackReportingExportSamples(
            split_exclusions=['test']
        ),
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackReportingRecordHistory(),
    ]


def get_image(batch):
    return 2 * batch['images'] - 1


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = trw.layers.convs_2d(
            input_channels=1,
            channels=[64, 128, 256, 2],
            convolution_kernels=[4, 4, 4, 3],
            strides=[2, 2, 2, 1],
            pooling_size=None,
            with_flatten=False,
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            last_layer_is_output=True
        )

    def forward(self, batch, image, is_real):
        o = self.model(image)
        o = global_average_pooling_2d(o)
        o_expected = int(is_real) * torch.ones(len(image), device=image.device, dtype=torch.long).unsqueeze(1)
        batch['o_expected'] = o_expected

        return {
            'classification': OutputClassification(
                o, o_expected,
                criterion_fn=LossMsePacked,  # LSGan loss function
            )
        }


class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = trw.layers.ConvsTransposeBase(
            2,
            input_channels=latent_size,
            channels=[256, 128, 64, 1],
            convolution_kernels=4,
            strides=[1, 2, 2, 2],
            paddings=[0, 1, 1, 1],
            activation=functools.partial(nn.LeakyReLU, negative_slope=0.2),
            squash_function=torch.tanh,
            target_shape=[28, 28]
        )

    def forward(self, batch, latent):
        latent = latent.unsqueeze(2).unsqueeze(2)
        o = self.model(latent)
        return o, collections.OrderedDict([
            ('image', OutputEmbedding(o)),
        ])


def create_model():
    latent_size = 64

    generator = Generator(latent_size=latent_size)
    discriminator = Discriminator()

    optimizer_fn = functools.partial(torch.optim.Adam, lr=0.001, betas=(0.5, 0.999))

    model = Gan(
        discriminator=discriminator,
        generator=generator,
        latent_size=latent_size,
        optimizer_discriminator_fn=optimizer_fn,
        optimizer_generator_fn=optimizer_fn,
        real_image_from_batch_fn=get_image
    )

    return model


options = trw.train.Options(num_epochs=50)
trainer = trw.train.TrainerV2(
    callbacks_per_epoch=per_epoch_callbacks(),
)

trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(batch_size=256, normalize_0_1=True),
    eval_every_X_epoch=1,
    model=create_model(),
    log_path='mnist_dcgan2',
    optimizers_fn=None  # the module has its own optimizers
)

