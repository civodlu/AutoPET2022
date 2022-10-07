import functools

import trw
import torch
import torch.nn as nn
from trw.layers import NormType


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch):
        assert len(batch['name_text']) == 1, 'expect only one sample!'
        i = batch['name']
        assert i.shape[0] == 1
        i = i[0]

        hidden = self.initHidden(i.device)
        for n in range(i.shape[0]):
            output, hidden = self._forward(i[n], hidden)
        return {
            'classification': trw.train.OutputClassification(
                output, batch['category_id'], classes_name='category_id')
        }

    def _forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, self.hidden_size, device=device)


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super().__init__()
        self.model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, bidirectional=bidirectional)

        if bidirectional:
            linear_in = 2 * hidden_size
        else:
            linear_in = hidden_size
        self.linear = trw.layers.denses(
            [linear_in, output_size],
            normalization_type=NormType.InstanceNorm,
            last_layer_is_output=True
        )

    def forward(self, batch):
        i = batch['name']
        outputs, _ = self.model(i.squeeze(0))
        cs = self.linear(outputs[-1])
        return {
            'classification': trw.train.OutputClassification(cs, batch['category_id'], classes_name='category_id')
        }


def create_model(model_type):
    if model_type == 'rnn':
        return RNN(input_size=57, hidden_size=128, output_size=18)
    elif model_type == 'lstm':
        return Lstm(input_size=57, hidden_size=128, output_size=18)
    else:
        raise NotImplementedError(f'model_type={model_type} unknown!')


# configure and run the training/evaluation
model_type = 'lstm'
options = trw.train.Options(num_epochs=10)
trainer = trw.train.TrainerV2()

results = trainer.fit(
    options,
    datasets=trw.datasets.create_name_nationality_dataset(),
    log_path='name_nationality_rnn',
    model=create_model(model_type=model_type),
    optimizers_fn=lambda datasets, model: trw.train.create_adam_optimizers_fn(
        datasets=datasets, model=model, learning_rate=0.001))

print('DONE!')
