import torch
import trw
from torch import nn
from trw.layers import AutoencoderConvolutionalVariational, crop_or_pad_fun
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        z_size = 50
        activation = nn.LeakyReLU

        encoder = trw.layers.ConvsBase(
            dimensionality=2,
            input_channels=1,
            channels=[16, 32, 64],
            convolution_kernels=[5, 5, 5],
            strides=2,
            pooling_size=None,
            activation=activation
        )

        decoder = trw.layers.ConvsTransposeBase(
            dimensionality=2,
            input_channels=z_size,
            channels=[32, 16, 8, 1],
            strides=[3, 2, 2, 2],
            convolution_kernels=[5, 5, 5, 5],
            last_layer_is_output=True,
            # make sure the decoded image is within [0..1], required for
            # the binary cross entropy loss
            squash_function=torch.sigmoid,
            activation=activation,
            paddings=0
        )

        self.autoencoder = AutoencoderConvolutionalVariational([1, 1, 28, 28], encoder, decoder, z_size)

    def forward(self, batch):
        images = batch['images']
        recon, mu, logvar = self.autoencoder.forward(images)
        random_samples = self.autoencoder.sample(len(images))
        random_samples = crop_or_pad_fun(random_samples, images.shape[2:])

        loss = self.autoencoder.loss_function(recon, images, mu, logvar, kullback_leibler_weight=0.1)
        return {
            'loss': trw.train.OutputLoss(loss),
            'recon': trw.train.OutputEmbedding(recon),
            'random_samples': trw.train.OutputEmbedding(random_samples)
        }


def per_epoch_fn():
    callbacks = [
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackSkipEpoch(
            nb_epochs=1,
            callbacks=[trw.callbacks.CallbackReportingExportSamples(table_name='random_samples', max_samples=5, split_exclusions=['train'])]),
    ]

    return callbacks


def pos_training_fn():
    return [
        trw.callbacks.CallbackReportingExportSamples(max_samples=500),
        trw.callbacks.CallbackSaveLastModel()
    ]


options = trw.train.Options(num_epochs=200)
trainer = trw.train.TrainerV2(
    callbacks_per_epoch=per_epoch_fn(),
    callbacks_post_training=pos_training_fn(),
)

model = Net()
results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True, batch_size=1024),
    log_path='mnist_autoencoder_variational',
    model=model,
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.001, step_size=120, gamma=0.1))


model.training = False
nb_images = 40
generated = model.autoencoder.decoder(torch.randn([nb_images, 50, 1, 1], dtype=torch.float32, device=trw.train.get_device(model)))

fig, axes = plt.subplots(nrows=1, ncols=nb_images, figsize=(nb_images, 2.5), sharey=True)
decoded_images = crop_or_pad_fun(generated, [28, 28])
image_width = decoded_images.shape[2]

for ax, img in zip(axes, decoded_images):
    curr_img = img.detach().to(torch.device('cpu'))
    ax.imshow(curr_img.view((image_width, image_width)), cmap='binary')

plt.show()
