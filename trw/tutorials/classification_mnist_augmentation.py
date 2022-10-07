import trw
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    """
    Defines our model for MNIST
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 2)
        self.fc1 = nn.Linear(20 * 6 * 6, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, batch):
        # a batch should be a dictionary of features
        x = batch['images'] / 255.0
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 20 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # here we create a softmax output that will use
        # the `targets` feature as classification target
        return {
            'softmax': trw.train.OutputClassification(x, batch['targets'], classes_name='targets')
        }


if __name__ == '__main__':
    # configure and run the training/evaluation
    options = trw.train.Options(num_epochs=40)
    trainer = trw.train.TrainerV2()
    
    # perform augmentation on each batch of training data
    transforms = [
        trw.transforms.TransformRandomCropPad(padding=[0, 2, 2])
    ]
    
    results = trainer.fit(
        options,
        datasets=trw.datasets.create_mnist_dataset(transforms=transforms, nb_workers=2),
        log_path='mnist_cnn_augmentation',
        model=Net(),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.1)
    )
    
    # calculate statistics of the final epoch
    output = results.outputs['mnist']['test']['softmax']
    accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
    assert accuracy >= 0.95
