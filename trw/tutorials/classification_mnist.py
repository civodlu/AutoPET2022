import torch.nn as nn
import numpy as np
import trw
from trw.train.outputs_trw import OutputClassification


class Net(nn.Module):
    """
    Defines our model for MNIST
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 20, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(20 * 6 * 6, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, batch):
        # a batch should be a dictionary of features
        x = self.model(batch['images'])

        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'classification': OutputClassification(x, batch['targets'])
        }


# configure and run the training/evaluation
options = trw.train.Options(num_epochs=40)
trainer = trw.train.TrainerV2()

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True),
    log_path='mnist_cnn',
    model=Net(),
    optimizers_fn=trw.train.OptimizerSGD(learning_rate=0.1))

# calculate statistics of the final epoch
output = results.outputs['mnist']['test']['classification']
accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
assert accuracy >= 0.95
