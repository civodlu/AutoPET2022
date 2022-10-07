import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trw.utils
from trw.train.outputs_trw import OutputClassificationBinary


class Net(nn.Module):
    """
    Defines our model for MNIST
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images']

        targets = (batch['targets'] == 6).long()

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = trw.utils.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': OutputClassificationBinary(x, targets)
        }


# configure and run the training/evaluation
options = trw.train.Options(num_epochs=40)
trainer = trw.train.TrainerV2()

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True),
    log_path='mnist_cnn_6_vs_all',
    model=Net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
        datasets=datasets, model=model, learning_rate=0.1))

# calculate statistics of the final epoch
output = results.outputs['mnist']['test']['softmax']
accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
assert accuracy >= 0.95
