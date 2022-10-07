import torch
from trw.train import OutputEmbedding, OutputSegmentation
import trw
import torch.nn as nn


class UNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_input = nn.InstanceNorm3d(1)
        self.model = trw.layers.UNetBase(3, 1, channels=[64, 128, 256], output_channels=3)

    def forward(self, batch):
        # a batch should be a dictionary of features
        labels = batch['label_voxels']
        x = batch['image_voxels']
        x = self.norm_input(x)

        o = self.model(x)

        x_2d = x[:, :, x.shape[2] // 2]
        x_2d = trw.transforms.resize(x_2d, (64, 64))

        o_2d = o[:, :, o.shape[2] // 2]
        o_2d = trw.transforms.resize(o_2d, (64, 64))

        labels_2d = labels[:, :, o.shape[2] // 2]
        labels_2d = trw.transforms.resize(labels_2d, (64, 64))

        return {
            'x_2d': OutputEmbedding(x_2d),
            'o_2d': OutputEmbedding(nn.Sigmoid()(o_2d)),
            'labels_2d': OutputEmbedding(labels_2d),
            'softmax': OutputSegmentation(o, labels)
        }


if __name__ == '__main__':
    nb_epochs = 400
    options = trw.train.Options(num_epochs=nb_epochs, device=torch.device('cuda:1'))
    trainer = trw.train.TrainerV2(
        callbacks_pre_training=[
            trw.callbacks.CallbackReportingStartServer(),
            trw.callbacks.CallbackReportingModelSummary(),
            trw.callbacks.CallbackReportingDatasetSummary(),
            trw.callbacks.CallbackReportingAugmentations(),
        ],
        callbacks_per_epoch=[
            trw.callbacks.CallbackLearningRateRecorder(),
            trw.callbacks.CallbackEpochSummary(),
            trw.callbacks.CallbackReportingRecordHistory(),
            trw.callbacks.CallbackReportingBestMetrics(),
            trw.callbacks.CallbackSkipEpoch(10, [
                trw.callbacks.CallbackReportingExportSamples(table_name='current_samples'),
            ], include_epoch_zero=True)
        ]
    )

    results = trainer.fit(
        options,
        datasets=trw.datasets.create_decathlon_dataset(
            'Task04_Hippocampus',
            transform_train=trw.transforms.TransformResizeModuloCropPad(multiple_of=8, mode='pad'),
            transform_valid=trw.transforms.TransformResizeModuloCropPad(multiple_of=8, mode='pad'),
            remove_patient_transform=True,
        ),
        log_path='decathlon_task4',
        model=UNetSegmentation(),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.01,
            step_size=nb_epochs // 6,
            gamma=0.2,
            weight_decay=2e-5,
            nesterov=True
         )
    )

    print('DONE!')

