import trw
import torch.nn as nn
import numpy as np
import torch


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch):
        # Set initial hidden and cell states
        images = batch['images'].reshape((-1, 28, 28))

        h0 = torch.zeros(self.num_layers, len(images), self.hidden_size).to(images.device)
        c0 = torch.zeros(self.num_layers, len(images), self.hidden_size).to(images.device)

        # Forward propagate LSTM
        out, _ = self.lstm(images, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return {
            'softmax': trw.train.OutputClassification(out, batch['targets'], classes_name='targets')
        }


options = trw.train.Options(num_epochs=150)
trainer = trw.train.TrainerV2()

results = trainer.fit(
    options,
    datasets=trw.datasets.create_mnist_dataset(normalize_0_1=True),
    log_path='mnist_rnn',
    model=RNN(28, 256, 2, 10),
    optimizers_fn=lambda datasets, m: trw.train.create_adam_optimizers_fn(
        datasets=datasets, model=m, learning_rate=0.005))

# calculate statistics of the final epoch
output = results.outputs['mnist']['test']['softmax']
accuracy = float(np.sum(output['output'] == output['output_truth'])) / len(output['output_truth'])
assert accuracy >= 0.95
