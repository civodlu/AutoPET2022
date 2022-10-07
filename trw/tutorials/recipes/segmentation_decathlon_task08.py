import copy
import functools
from typing import List, Callable
import numpy as np
import torch
from trw.basic_typing import Batch, Tensor, TorchTensorNCX, Activation, TensorNCX, ShapeCX
from trw.layers import ModuleWithIntermediate, UNetBase, LayerConfig, default_layer_config
from trw.layers.blocks import ConvBlockType, BlockConvNormActivation
from trw.train import OutputEmbedding, OutputSegmentation, Output, get_device
import trw
import torch.nn as nn
from trw.transforms import random_fixed_geometry_within_geometries, SpatialInfo, resize
from trw.utils import to_value
from typing_extensions import Literal


class MultiScaleLossUnet(nn.Module):
    """
    Apply a multi-scale loss on each `up` layer of a UNet to help propagate the
    gradient through the layers.
    """
    def __init__(
            self,
            unet: UNetBase,
            input_target_shape: ShapeCX,
            output_creator: Callable[[TorchTensorNCX, TensorNCX], Output],
            output_block: ConvBlockType = BlockConvNormActivation,
            discard_top_k_outputs: int = 1,
            resize_mode: Literal['nearest', 'linear'] = 'nearest',
            config: LayerConfig = default_layer_config(dimensionality=None)):

        super().__init__()
        self.unet = unet
        assert len(input_target_shape) == unet.dim + 1, 'must be a shape with `N` component removed!'

        device = get_device(unet)
        self.discard_top_k_outputs = discard_top_k_outputs

        # dummy test to get the intermediate layer shapes
        dummy_input = torch.zeros([1] + list(input_target_shape), device=device)
        outputs = unet.forward_with_intermediate(dummy_input)

        # no activation, these are all output nodes!
        config = copy.copy(config)
        config.set_dim(unet.dim)
        config.activation = None
        config.norm = None

        self.outputs = nn.ModuleList()
        for o in outputs[discard_top_k_outputs:]:
            output = output_block(
                config,
                o.shape[1],
                self.unet.output_channels)
            self.outputs.append(output)

        self.output_creator = output_creator
        self.resize_mode = resize_mode

    def forward(self, x, target, latent=None):
        os = self.unet.forward_with_intermediate(x, latent=latent)

        outputs = []
        for n, o in enumerate(os[self.discard_top_k_outputs:]):
            # reshape the output to have the expected target channels
            o_tfm = self.outputs[n](o)
            target_resized = resize(target, o.shape[2:], mode=self.resize_mode)
            output = self.output_creator(o_tfm, target_resized)
            outputs.append(output)

        return outputs


class UNetSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_input = nn.InstanceNorm3d(1)
        self.model = trw.layers.UNetBase(3, 1, channels=[32, 64, 128, 256, 320], output_channels=3)
        self.multi_scale_loss = MultiScaleLossUnet(self.model, [1, 256, 256, 64], output_creator=trw.train.OutputSegmentation)

    def forward(self, batch):
        # a batch should be a dictionary of features
        labels = batch['label_voxels']
        x = batch['image_voxels']
        x = (torch.clamp(x, -3.0, 243.0) - 104.37) / 52.62  # https://arxiv.org/pdf/1904.08128.pdf

        multiscale_o = self.multi_scale_loss(x, labels)
        o = multiscale_o[-1].output

        x_2d = x[:, :, x.shape[2] // 2]
        o_2d = o[:, :, o.shape[2] // 2]
        labels_2d = labels[:, :, o.shape[2] // 2]

        outputs = {
            'x_2d': OutputEmbedding(x_2d),
            'o_2d': OutputEmbedding(nn.Sigmoid()(o_2d)),
            'labels_2d': OutputEmbedding(labels_2d),
            #'softmax': OutputSegmentation(o, labels)
        }
        for i, o in enumerate(multiscale_o):
            outputs[f'softmax_{i}'] = o
        return outputs


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
            trw.callbacks.CallbackReportingModelSummary(),
            #trw.callbacks.CallbackReportingDatasetSummary(),
            #trw.callbacks.CallbackReportingAugmentations(),
        ],
        callbacks_per_epoch_fn=lambda: [
            trw.callbacks.CallbackLearningRateRecorder(),
            trw.callbacks.CallbackEpochSummary(),
            trw.callbacks.CallbackReportingRecordHistory(),
            trw.callbacks.CallbackReportingBestMetrics(),
            trw.callbacks.CallbackSkipEpoch(10, [
                trw.callbacks.CallbackReportingExportSamples(table_name='current_samples'),
            ], include_epoch_zero=False)
        ]
    )

    transform = trw.transforms.TransformResample(
        resampling_geometry=functools.partial(random_fixed_geometry_within_geometries,
                                              fixed_geometry_shape=[256, 256, 64],
                                              fixed_geometry_spacing=[1.5, 1.5, 1.5]),
        get_spatial_info_from_batch_name=get_spatial_info_type,
    )

    results = trainer.fit(
        options,
        eval_every_X_epoch=10,
        inputs_fn=lambda: trw.datasets.create_decathlon_dataset(
            'Task08_HepaticVessel',
            transform_train=transform,
            transform_valid=transform,
        ),
        run_prefix='decathlon_task8',
        model_fn=lambda options: UNetSegmentation(),
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            #learning_rate=0.005,
            learning_rate=0.001,
            step_size=nb_epochs // 6,
            gamma=0.2
        ))

    print('DONE!')

