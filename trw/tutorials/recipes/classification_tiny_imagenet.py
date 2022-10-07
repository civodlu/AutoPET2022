import collections
import functools

import torch
import trw
import torch.nn as nn
import torchvision


class SimpleNet2(nn.Module):
    def __init__(self, options, conv_filters=[64, 128, 256, 200], conv_repeats=[3, 3, 3, 1]):
        super().__init__()

        activation_conv = nn.LeakyReLU

        self.convs = trw.layers.convs_2d(
            input_channels=3,
            channels=conv_filters,
            activation=activation_conv,
            convolution_repeats=conv_repeats,
            convolution_kernels=3,
            last_layer_is_output=True,
        )

        self.global_pooling = nn.AvgPool2d(kernel_size=[4, 4])

    def forward(self, batch):
        images = batch['images'].float() / 255.0
        r = self.convs(images)
        r = self.global_pooling(r).squeeze()
        assert len(r.shape) == 2
        return collections.OrderedDict([
            ('classification', trw.train.OutputClassification(r, batch['targets'], classes_name='targets'))
        ])


class Torchvision(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet34(num_classes=200)

    def forward(self, batch):
        images = batch['images'].float() / 255.0
        r = self.model.forward(images)
        return collections.OrderedDict([
            ('classification', trw.train.OutputClassification(r, batch['targets'], classes_name='targets'))
        ])


def per_epoch_additional_callbacks():
    return [
        trw.callbacks.CallbackSkipEpoch(20, [
            trw.callbacks.CallbackReportingExportSamples(max_samples=20, reporting_scatter_x='split_name', table_name='errors'),
            trw.callbacks.CallbackSaveLastModel(model_name='interim', with_outputs=False),
        ], include_epoch_zero=True),
    ]


if __name__ == '__main__':
    transforms_train = [
        trw.transforms.TransformCast(feature_names=['images'], cast_type='float'),
        trw.transforms.TransformRandomCutout(
            cutout_size=functools.partial(trw.transforms.cutout_random_size, min_size=[3, 5, 5], max_size=[3, 15, 15]),
            cutout_value_fn=trw.transforms.cutout_random_ui8_torch),
        trw.transforms.TransformRandomCropPad(padding=[0, 4, 4]),
        trw.transforms.TransformRandomFlip(axis=3),
    ]

    transforms_valid = [
        trw.transforms.TransformCast(feature_names=['images'], cast_type='float'),
    ]

    options = trw.train.Options(num_epochs=2000, device=torch.device('cuda:1'))
    trainer = trw.train.TrainerV2(
        callbacks_per_epoch=trw.train.default_per_epoch_callbacks(
            additional_callbacks=per_epoch_additional_callbacks())
    )
    results = trainer.fit(
        options,
        datasets=trw.datasets.create_tiny_imagenet_dataset(
            batch_size=64,
            num_images_per_class=500,
            transforms_train=transforms_train,
            transforms_valid=transforms_valid),
        log_path='tiny_imagenet',
        model=SimpleNet2(options),

        eval_every_X_epoch=1,
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.01,
            momentum=0.9,
            weight_decay=5e-4,
            step_size=100,
            gamma=0.3,
            nesterov=True
        )
    )
