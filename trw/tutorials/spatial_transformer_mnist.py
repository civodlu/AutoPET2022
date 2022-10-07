"""
Based on the "Spatial Transformer Networks", https://arxiv.org/pdf/1506.02025.pdf

Use a transformer network to locate within a cluttered image a region of interest. The classification accuracy
will drive the training of the transformer network.
"""
import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


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
        return x[:, :, 0:24, 0:24]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = trw.layers.convs_2d(1, channels=[8, 16], convolution_kernels=7, strides=2)
        self.denses = trw.layers.denses([16 * 1 * 1, 500, 10], last_layer_is_output=True)

        self.transformer = NetSpatialTransformer()

    def forward(self, batch):
        x = batch['images'] / 255.0
        aligned_x = self.transformer(x)
        x = self.convs(aligned_x)
        x = self.denses(x)

        return {
            'softmax': trw.train.OutputClassification(x, batch['targets'], classes_name='targets'),
            'aligned': trw.train.OutputEmbedding(aligned_x)
        }


def per_epoch_fn():
    callbacks = [
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackSkipEpoch(
            nb_epochs=10,
            callbacks=[trw.callbacks.CallbackReportingExportSamples(
                table_name='current_samples',
                max_samples=5,
                split_exclusions=['train'])]),
    ]

    return callbacks


# configure and run the training/evaluation
options = trw.train.Options(num_epochs=400)
trainer = trw.train.TrainerV2(
    callbacks_per_epoch=per_epoch_fn(),
)

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_cluttered_datasset(cluttered_size=(64, 64)),
    log_path='mnist_cluttered_spatial_transformer',
    model=Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.00051, step_size=100, gamma=0.1)
)

# calculate statistics of the final epoch
output = results.outputs['mnist']['test']['softmax']
accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
assert accuracy >= 0.95
