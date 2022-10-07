import functools
from typing import List

import torch
from trw.train import OutputEmbedding, OutputSegmentation
import trw
import torch.nn as nn
import monai.networks.nets
from trw.transforms import SpatialInfo, random_fixed_geometry_within_geometries
from trw.basic_typing import Batch, Tensor
from trw.utils import to_value
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.norm_input = nn.InstanceNorm3d(1)
        #self.model = monai.networks.nets.UNet(3, 1, 2, channels=[64, 128, 256, 512], strides=[2, 2, 2, 1])
        self.model = trw.layers.UNetBase(
            3,
            1,
            channels=[16, 32, 64, 128, 256],
            output_channels=2
        )

    def forward(self, batch):
        # a batch should be a dictionary of features
        labels = batch['label_voxels']
        x = batch['image_voxels']
        #x = self.norm_input(x)

        x = (torch.clamp(x, -30.0, 162.82) - 62.18) / 32.65  # https://arxiv.org/pdf/1904.08128.pdf
        o = self.model(x)

        x_2d = x[:, :, x.shape[2] // 2]
        x_2d = trw.transforms.resize(x_2d, (64, 64))

        o_2d = o[:, :, o.shape[2] // 2]
        o_2d = trw.transforms.resize(o_2d, (64, 64))
        o_2d = o_2d[:, 1].unsqueeze(1)

        labels_2d = labels[:, :, o.shape[2] // 2]
        labels_2d = trw.transforms.resize(labels_2d, (64, 64))

        return {
            'x_2d': OutputEmbedding(x_2d),
            'o_2d': OutputEmbedding(nn.Sigmoid()(o_2d)),
            'labels_2d': OutputEmbedding(labels_2d),
            'softmax': OutputSegmentation(o, labels)
        }


def get_spacing_from_4x4(tfm: Tensor) -> List[float]:
    assert tfm.shape == (4, 4)
    tfm = to_value(tfm)
    spacing = [np.linalg.norm(tfm[0:3, n]) for n in range(3)]
    return spacing


def get_translation_from_4x4(tfm: Tensor) -> np.ndarray:
    assert tfm.shape == (4, 4)
    tfm = to_value(tfm)
    return tfm[0:3, 3]


def get_spatial_info_type(batch: Batch, name: str) -> SpatialInfo:
    v = batch[name]
    assert len(v.shape) == 5
    assert v.shape[0] == 1
    assert v.shape[1] == 1

    affine_matrix = batch[name.replace('_voxels', '_affine')]
    assert len(affine_matrix.shape) == 3 and affine_matrix.shape[0] == 1
    affine_matrix = affine_matrix[0]

    spacing = get_spacing_from_4x4(affine_matrix)
    origin = get_translation_from_4x4(affine_matrix)
    return SpatialInfo(
        origin=origin,
        spacing=spacing,
        shape=v.shape[2:]
    )


if __name__ == '__main__':
    nb_epochs = 400
    options = trw.train.Options(num_epochs=nb_epochs)
    trainer = trw.train.Trainer(
        callbacks_pre_training_fn=lambda: [
            trw.callbacks.CallbackReportingStartServer(),
            #trw.callbacks.CallbackReportingModelSummary(),
            #trw.callbacks.CallbackReportingDatasetSummary(),
            #trw.callbacks.CallbackReportingAugmentations(),
        ],
        callbacks_per_epoch_fn=lambda: [
            #trw.callbacks.CallbackLearningRateRecorder(),
            trw.callbacks.CallbackEpochSummary(),
            trw.callbacks.CallbackReportingRecordHistory(),
            #trw.callbacks.CallbackReportingBestMetrics(),
            trw.callbacks.CallbackSkipEpoch(10, [
                trw.callbacks.CallbackReportingExportSamples(table_name='current_samples'),
            ], include_epoch_zero=False)
        ]
    )

    transform = trw.transforms.TransformResample(
        resampling_geometry=functools.partial(random_fixed_geometry_within_geometries,
            fixed_geometry_shape=[256, 256, 96],
            fixed_geometry_spacing=[1.55, 1.55, 3.09]),
        get_spatial_info_from_batch_name=get_spatial_info_type,
    )

    results = trainer.fit(
        options,
        eval_every_X_epoch=20,
        inputs_fn=lambda: trw.datasets.create_decathlon_dataset(
            'Task10_Colon',
            transform_train=transform,
            transform_valid=transform,
        ),
        run_prefix='decathlon_task10',
        model_fn=lambda options: Net(),
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.001,
            step_size=nb_epochs // 6,
            gamma=0.2
        ))

    print('DONE!')

