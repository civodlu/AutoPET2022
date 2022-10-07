import torch.nn as nn
import torch
import trw
from trw.layers import crop_or_pad_fun, AutoencoderConvolutionalVariational, \
    AutoencoderConvolutionalVariationalConditional
import matplotlib.pyplot as plt
from trw.train.losses import one_hot


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        z_filters_in = 128

        self.latent_size = 40
        self.y_size = 10

        activation = nn.LeakyReLU

        encoder = trw.layers.ConvsBase(
            dimensionality=2,
            input_channels=1,
            channels=[16, 32, 64, z_filters_in],
            convolution_kernels=[5, 5, 5, 5],
            strides=2,
            pooling_size=None,
            activation=activation
        )

        decoder = trw.layers.ConvsTransposeBase(
            dimensionality=2,
            input_channels=self.latent_size + self.y_size,
            channels=[64, 32, 16, 1],
            strides=[1, 1, 1, 2],
            convolution_kernels=[5, 5, 5, 5],
            last_layer_is_output=True,
            # make sure the decoded image is within [0..1], required for
            # the binary cross entropy loss
            squash_function=torch.sigmoid,
            activation=activation,
            paddings=0
        )

        self.autoencoder = AutoencoderConvolutionalVariationalConditional([1, 1, 28, 28], encoder, decoder, self.latent_size, self.y_size)

    def forward(self, batch):
        images = batch['images']
        y = batch['targets']
        device = trw.train.get_device(self)
        y_one_hot = one_hot(y, self.y_size).to(device)
        recon, mu, logvar = self.autoencoder.forward(images, y_one_hot)

        with torch.no_grad():
            random_recon_y = one_hot(y, self.y_size)  # keep same distribution as data
            random_recon = self.autoencoder.sample_given_y(random_recon_y)

        loss = AutoencoderConvolutionalVariational.loss_function(recon, images, mu, logvar, kullback_leibler_weight=0.1)
        return {
            'loss': trw.train.OutputLoss(loss),
            'recon': trw.train.OutputEmbedding(recon),
            'random_recon': trw.train.OutputEmbedding(random_recon)
        }


def per_epoch_fn():
    callbacks = [
        trw.callbacks.CallbackEpochSummary(),
        trw.callbacks.CallbackSkipEpoch(
            nb_epochs=10,
            callbacks=[trw.callbacks.CallbackReportingExportSamples(table_name='random_samples', max_samples=5, split_exclusions=['train'])]),
    ]

    return callbacks


def pos_training_fn():
    return [
        trw.callbacks.CallbackReportingExportSamples(max_samples=1000),
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
    log_path='mnist_autoencoder_variational_conditional',
    model=model,
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.001, step_size=120, gamma=0.1))


model.training = False
nb_images = 40

device = trw.train.get_device(model)
latent = torch.randn([nb_images, model.latent_size], device=device)
y = one_hot(torch.ones([nb_images], dtype=torch.long, device=device) * 7, 10)
latent_y = torch.cat([latent, y], dim=1)
latent_y = latent_y.view(latent_y.shape[0], latent_y.shape[1], 1, 1)
generated = model.autoencoder.decoder(latent_y)

fig, axes = plt.subplots(nrows=1, ncols=nb_images, figsize=(nb_images, 2.5), sharey=True)
decoded_images = crop_or_pad_fun(generated, [28, 28])
image_width = decoded_images.shape[2]

for ax, img in zip(axes, decoded_images):
    curr_img = img.detach().to(torch.device('cpu'))
    ax.imshow(curr_img.view((image_width, image_width)), cmap='binary')

plt.show()
