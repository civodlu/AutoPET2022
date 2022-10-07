"""
Based on the "Spatial Transformer Networks", https://arxiv.org/pdf/1506.02025.pdf

Use a transformer network to locate within a cluttered image a region of interest. Here we do not use the actual
classification of the digits, simply that they belong to the same category. The training will sample triplets
so that sample 0 and 1 belong to the same class and sample 2 is a different class.
"""
import trw
import torch.nn as nn
import torch.nn.functional as F
import torch
import collections


class NetSpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization_convs = trw.layers.convs_2d(1, channels=[16, 32], convolution_kernels=7, strides=2, dropout_probability=0.5)
        self.localization_regressor = trw.layers.denses([32 * 4 * 4, 64, 2 * 3], last_layer_is_output=True, dropout_probability=0.5)

        # make sure the initial transformation is reasonable (e.g., identity)
        self.localization_regressor[-1].weight.data.zero_()
        self.localization_regressor[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, x):
        xs = self.localization_convs(x)
        xs = self.localization_regressor(xs)
        theta = xs.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x[:, :, 0:32, 0:32]


class NetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = trw.layers.convs_2d(1, channels=[8, 16], convolution_kernels=7, strides=2)
        self.denses = trw.layers.denses([16 * 2 * 2, 64, 16], last_layer_is_output=True)
        self.transformer = NetSpatialTransformer()

    def forward(self, images):
        aligned_x = self.transformer(images)
        x = self.convs(aligned_x)
        x = self.denses(x)
        return x, aligned_x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = NetEmbedding()

    def forward(self, batch):
        targets = batch['targets']
        anchor, positive, negative = trw.train.make_triplet_indices(targets)
        images = batch['images'] / 255.0

        samples = images[anchor]
        samples_positive = images[positive]
        samples_negative = images[negative]

        samples_embedding, aligned_samples = self.embedding(samples)
        samples_positive_embedding, aligned_samples_positive = self.embedding(samples_positive)
        samples_negative_embedding, aligned_samples_negative = self.embedding(samples_negative)

        return collections.OrderedDict([
            ('triplets', trw.train.OutputTriplets(
                samples_embedding,
                samples_positive_embedding,
                samples_negative_embedding)),
            ('aligned_samples', trw.train.OutputEmbedding(aligned_samples)),
            ('aligned_samples_positive', trw.train.OutputEmbedding(aligned_samples_positive)),
            ('aligned_samples_negative', trw.train.OutputEmbedding(aligned_samples_negative)),
            ('samples', trw.train.OutputEmbedding(samples)),
            ('samples_positive', trw.train.OutputEmbedding(samples_positive)),
            ('samples_negative', trw.train.OutputEmbedding(samples_negative)),
        ])


def per_epoch_fn():
    callbacks = [
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackSkipEpoch(
            nb_epochs=10,
            callbacks=[trw.callbacks.CallbackReportingExportSamples(table_name='random_samples', max_samples=5, split_exclusions=['train'])]),
    ]

    return callbacks


def pre_training_fn():
    callbacks = [
        trw.callbacks.CallbackReportingStartServer(),
        trw.callbacks.CallbackReportingExportSamples(table_name='random_samples', max_samples=5, split_exclusions=['train']),
    ]
    return callbacks


# configure and run the training/evaluation
options = trw.train.Options(num_epochs=400)
trainer = trw.train.TrainerV2(
    callbacks_pre_training=pre_training_fn(),
    callbacks_per_epoch=per_epoch_fn(),
)

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_cluttered_datasset(cluttered_size=(64, 64), nb_workers=0),
    log_path='mnist_cluttered_spatial_transformer_triplets',
    model=Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
        datasets=datasets,
        model=model,
        learning_rate=0.001,
        step_size=200,
        gamma=0.1)
)
