import os
# we already use worker threads, limit each process to 1 thread!
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from trw.layers.efficient_net import EfficientNet


import trw
import numpy as np

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # base image is 224x224 for efficient net, but cifar10 is 32x32, so rescale the images
        self.net = EfficientNet(
            dimensionality=2,
            input_channels=3,
            output_channels=10,
        )

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images']
        x = self.net(x)

        return {
            'softmax': trw.train.OutputClassification(x, batch['targets'], classes_name='targets')
        }


def create_model():
    model = Net()
    model = trw.train.DataParallelExtended(model)
    return model


if __name__ == '__main__':
    # configure and run the training/evaluation
    assert torch.cuda.device_count() >= 2, 'not enough CUDA devices for this multi-GPU tutorial!'
    options = trw.train.Options(num_epochs=200)
    trainer = trw.train.TrainerV2(
        callbacks_post_training=None,
        callbacks_pre_training=None,
    )

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    transform_train = [
        trw.transforms.TransformRandomFlip(axis=3),
        trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 16), probability=0.2),
        trw.transforms.TransformRandomCropPad(padding=[0, 4, 4]),
        trw.transforms.TransformResize(size=[224, 224]),
        trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
    ]

    transform_valid = [
        trw.transforms.TransformResize(size=[224, 224]),
        trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
    ]

    datasets = trw.datasets.create_cifar10_dataset(
        transform_train=transform_train,
        transform_valid=transform_train, nb_workers=16,
        batch_size=100, data_processing_batch_size=50,
    )

    results = trainer.fit(
        options,
        datasets=datasets,
        log_path='cifar10_efficient_net_multigpu',
        model=create_model(),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_scheduler_step_lr_fn(
            datasets=datasets, model=model, learning_rate=0.05, momentum=0.9, weight_decay=0, step_size=50,
            gamma=0.3))

    # should converge to 94% accuracy
