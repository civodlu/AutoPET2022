import trw
import numpy as np


def create_net(options):
    n = trw.simple_layers.Input([None, 1, 28, 28], 'images')
    n = trw.simple_layers.Conv2d(n, out_channels=16, kernel_size=5, stride=2)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.MaxPool2d(n, 2, 2)
    n = trw.simple_layers.Flatten(n)
    n = trw.simple_layers.Linear(n, 500)
    n = trw.simple_layers.ReLU(n)
    n = trw.simple_layers.Linear(n, 10)
    n = trw.simple_layers.OutputClassification(n, output_name='softmax', classes_name='targets')
    return trw.simple_layers.compile_nn([n])


# configure and run the training/evaluation
trainer = trw.train.TrainerV2()

results = trainer.fit(
    trw.train.Options(num_epochs=10),
    datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True),
    log_path='mnist_cnn',
    model=create_net(),
    optimizers_fn=lambda datasets, model: trw.train.create_sgd_optimizers_fn(datasets=datasets, model=model, learning_rate=0.1))

# calculate statistics of the final epoch
output = results.outputs['mnist']['test']['softmax']
accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
assert accuracy >= 0.95
