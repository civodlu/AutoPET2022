if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    torch.multiprocessing.set_start_method('spawn')
    import trw
    import numpy as np

    import torch.nn as nn
    import logging

    logging.basicConfig(filename='log.log', filemode='w', level=logging.DEBUG)


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = trw.layers.PreActResNet18()

        def forward(self, batch):
            x = batch['images']
            x = self.net(x)
            return {
                'softmax': trw.train.OutputClassification(x, batch['targets'], classes_name='targets')
            }


    def create_model():
        model = Net()
        return model


    # configure and run the training/evaluation
    num_epochs = 200
    options = trw.train.Options(num_epochs=num_epochs)
    trainer = trw.train.TrainerV2(
        callbacks_post_training=None,
        callbacks_pre_training=None,
    )

    mean = np.asarray([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.asarray([0.2023, 0.1994, 0.2010], dtype=np.float32)

    transform_train = [
        trw.transforms.TransformRandomCropPad(padding=[0, 4, 4], mode='constant'),
        trw.transforms.TransformRandomFlip(axis=3),
        trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 16), probability=0.2),
        trw.transforms.TransformNormalizeIntensity(mean=mean, std=std),
        trw.transforms.TransformMoveToDevice(device=options.workflow_options.device, non_blocking=True)
    ]

    transform_valid = [
        trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
    ]

    datasets = trw.datasets.create_cifar10_dataset(
        transform_train=transform_train,
        transform_valid=transform_valid,
        nb_workers=2,
        batch_size=128,
        data_processing_batch_size=128
    )

    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    optimizer_fn_1 = lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets,
        model,
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False,
        scheduler_fn=scheduler_fn
    )

    model = Net()
    results = trainer.fit(
        options,
        datasets=datasets,
        log_path='cifar10_resnet',
        model=model,
        optimizers_fn=optimizer_fn_1
    )


