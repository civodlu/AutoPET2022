from functools import partial

import trw
import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_decoder = trw.layers.AutoencoderConvolutional(
            2,
            1,
            [8, 16, 32],
            [32, 16, 8, 1],  # make sure we are cropping the decoded output by adding another layer
            convolution_kernels=3,
            squash_function=torch.sigmoid,  # make sure the output is [0..1] domain
            last_layer_is_output=True  # do not apply the activation on the last layer
        )

    def forward(self, batch):
        x = batch['images']
        encoded_x, decoded_x = self.encoder_decoder.forward_with_intermediate(x)

        return {
            'regression': trw.train.OutputRegression(decoded_x, x),
        }


def per_epoch_fn():
    callbacks = [
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackSkipEpoch(
            nb_epochs=1,
            callbacks=[trw.callbacks.CallbackReportingExportSamples(table_name='random_samples', max_samples=5, split_exclusions=['train'])]),
    ]

    return callbacks


options = trw.train.Options(num_epochs=100)
trainer = trw.train.TrainerV2(
    callbacks_per_epoch=per_epoch_fn(),
    callbacks_post_training=[trw.callbacks.CallbackReportingExportSamples(max_samples=2000)],
)

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True),
    log_path='mnist_autoencoder',
    model=Net(),
    optimizers_fn=partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.25))

