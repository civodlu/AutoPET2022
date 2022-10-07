import trw
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torchvision
import collections


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
        x = batch['images']
        
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


def create_data():
    """
    Create the datasets using pytorch data loader
    """
    transforms = torchvision.transforms.ToTensor()
    batch_size = 1000
    num_workers = 4
    
    # first, check if we have some environment variables configured
    root = os.environ.get('TRW_DATA_ROOT')
    if root is None:
        root = './data'
    
    # standard pytorch pipeline using a data loader
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, transform=transforms, download=True)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    mnist_test = torchvision.datasets.MNIST(root=root, train=False, transform=transforms, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    # adapt to trw interface
    mnist = collections.OrderedDict()
    mnist['train'] = trw.train.SequenceAdaptorTorch(train_loader, features=['images', 'targets'])
    mnist['test'] = trw.train.SequenceAdaptorTorch(test_loader, features=['images', 'targets'])
    return {'mnist': mnist}
    

if __name__ == '__main__':
    # configure and run the training/evaluation
    options = trw.train.Options(num_epochs=40)
    trainer = trw.train.TrainerV2()
    
    results = trainer.fit(
        options,
        datasets=create_data(),
        log_path='mnist_cnn_dataloader',
        model=Net(),
        optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(
            datasets=datasets,
            model=model,
            learning_rate=0.1))
    
    # calculate statistics of the final epoch
    output = results.outputs['mnist']['test']['softmax']
    accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
    assert accuracy >= 0.95
