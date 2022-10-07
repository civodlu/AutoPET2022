import trw
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        convs = trw.layers.ConvsBase(dimensionality=2, input_channels=3, channels=[16, 32, 64])
        self.fcnn = trw.layers.FullyConvolutional(
            dimensionality=2,
            base_model=convs,
            input_channels=64,
            deconv_filters=[32, 16, 16],
            convolution_kernels=5,
            strides=2,
            nb_classes=2
        )

    def forward(self, batch):
        x = self.fcnn(batch['image'])

        return {
            'segmentation': trw.train.OutputSegmentation(output=x, output_truth=batch['mask']),
            'segmentation_output': trw.train.OutputEmbedding(x.argmax(dim=1, keepdim=True))
        }


def per_epoch_callbacks():
    return [
        trw.callbacks.CallbackSkipEpoch(5, [
                trw.callbacks.CallbackReportingExportSamples(max_samples=10),
        ]),
        trw.callbacks.CallbackEpochSummary(),
    ]


trainer = trw.train.TrainerV2(callbacks_per_epoch=per_epoch_callbacks())

results = trainer.fit(
    trw.train.Options(num_epochs=50),
    datasets=trw.datasets.create_fake_symbols_2d_dataset(
        nb_samples=1000,
        image_shape=[256, 256],
        nb_classes_at_once=1,
        batch_size=50,
        max_classes=5
    ),
    log_path='synthetic_segmentation_fcnn',
    model=Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.01, step_size=50, gamma=0.3))
