import trw
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = trw.layers.UNetBase(2, input_channels=3, output_channels=2, channels=[4, 8, 16])

    def forward(self, batch):
        x = self.unet(batch['image'])

        return {
            'segmentation': trw.train.OutputSegmentation(output=x, output_truth=batch['mask']),
            'segmentation_output': trw.train.OutputEmbedding(x.argmax(dim=1, keepdim=True))
        }


def per_epoch_callbacks():
    return [
        trw.callbacks.CallbackReportingExportSamples(max_samples=5),
        trw.callbacks.CallbackEpochSummary(),
    ]


trainer = trw.train.TrainerV2(callbacks_per_epoch=per_epoch_callbacks())

results = trainer.fit(
    trw.train.Options(num_epochs=15),
    datasets=trw.datasets.create_fake_symbols_2d_dataset(
        nb_samples=1000, image_shape=[256, 256], nb_classes_at_once=1, batch_size=50),
    log_path='synthetic_segmentation_unet',
    model=Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(
        datasets=datasets, model=model, learning_rate=0.05, step_size=50, gamma=0.3))
