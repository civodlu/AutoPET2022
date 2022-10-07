# we already use multiprocessing, multithreading will not improve anything here so disable it
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import torch.nn as nn
import torch
import trw


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = trw.layers.UNetBase(dim=2, input_channels=3, channels=[32, 64, 128, 256], output_channels=35)

    def forward(self, batch):
        x = batch['image']
        x = x.float()
        x = self.unet(x)

        return {
            'segmentation': trw.train.OutputSegmentation(x, batch['segmentation']),
            'mask': trw.train.OutputEmbedding(torch.sigmoid(x)),
        }


if __name__ == '__main__':
    trainer = trw.train.Trainer(callbacks_pre_training_fn=None)
    results = trainer.fit(
        trw.train.Options(num_epochs=200),
        inputs_fn=lambda: trw.datasets.create_cityscapes_dataset(batch_size=2, nb_workers=2),
        run_prefix='cityscapes_segmentation_unet',
        model_fn=lambda options: Net(),
        optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_scheduler_step_lr_fn(datasets=datasets, model=model, learning_rate=0.001, step_size=50, gamma=0.3))
