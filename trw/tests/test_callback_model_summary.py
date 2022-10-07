from unittest import TestCase
import trw.train
import torch
import torch.nn as nn
import warnings

from trw.callbacks.callback_reporting_model_summary import model_summary_base


class ModelDense(nn.Module):
    def __init__(self, n_output=5):
        super().__init__()

        self.d1 = nn.Linear(10, 100)
        self.d2 = nn.Linear(100, n_output)

    def forward(self, batch):
        return self.d2(self.d1(batch['input']))


class ModelDense_2_inputs(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = nn.Linear(10, 100)
        self.d2 = nn.Linear(10, 100)

    def forward(self, x1, x2):
        return self.d1(x1) + self.d2(x2)


class ModelDense_2_outputs(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = nn.Linear(10, 100)

    def forward(self, x1):
        return self.d1(x1), self.d1(x1)


class ModelDense_2_inputs_head(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = ModelDense_2_inputs()

    def forward(self, batch):
        return self.model(batch['input'], batch['input'])


class ModelDenseSequential(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 5)
        )

    def forward(self, batch):
        return self.d1(batch['input'])


class ModelNested(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = ModelDense()

    def forward(self, batch):
        return self.d1(batch)


class ModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, batch):
        # Set initial hidden and cell states
        images = batch['images'].reshape((-1, 28, 28))

        h0 = torch.zeros(self.num_layers, len(images), self.hidden_size, requires_grad=False).to(images.device)
        c0 = torch.zeros(self.num_layers, len(images), self.hidden_size, requires_grad=False).to(images.device)

        # Forward propagate LSTM
        out, _ = self.lstm(images, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return {
            'softmax': trw.train.OutputClassification(out, batch['targets'], classes_name='targets')
        }


class TestCallbackModelSummary(TestCase):
    def test_simple_model(self):
        model = ModelDense()
        batch = {
            'input': torch.zeros([11, 10])
        }

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_simple_model_internal_sequential(self):
        model = ModelDenseSequential()
        batch = {
            'input': torch.zeros([11, 10])
        }

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_simple_model_sequential(self):
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 5)
        )

        batch = torch.zeros([11, 10])

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_simple_model_nested(self):
        model = ModelNested()
        batch = {
            'input': torch.zeros([11, 10])
        }

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_simple_model_2_outputs(self):
        model = ModelDense_2_outputs()
        batch = {
            'input': torch.zeros([11, 10])
        }

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch['input'])

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_simple_model_nested_2heads(self):
        model = ModelDense_2_inputs_head()
        batch = {
            'input': torch.zeros([11, 10])
        }

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        expected_trainables = 2 * 10 * 100 + 2 * 100
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_rnn_model(self):
        model = ModelRNN(28, 256, 1, 10)
        batch = {
            'images': torch.zeros([32, 1, 28, 28]),
            'targets': torch.zeros([32], dtype=torch.int64),
        }

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        assert len(list(summary.values())[0]['input_shape']) == 3
        assert len(list(summary.values())[0]['output_shape']) == 3

        expected_trainables = 295434
        assert expected_trainables == trainable_params
        assert total_params == trainable_params

    def test_multi_gpus(self):
        """
        make sure we don't count the models replicated on different GPUs
        """
        nb_cuda_devices = torch.cuda.device_count()
        if nb_cuda_devices < 2:
            # we do not have enough GPUs, abot the test
            warnings.warn(f'This test can\'t be run. Requires CUDA devices=2, got={nb_cuda_devices}', ResourceWarning)
            return

        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 5)
        )

        device = torch.device('cuda:0')
        model = trw.train.DataParallelExtended(model).to(device)
        batch = torch.zeros([11, 10], device=device)

        summary, total_output_size, total_params_size, total_params, trainable_params = model_summary_base(model, batch)

        # 2 denses + 2 biases
        expected_trainables = 10 * 100 + 100 * 5 + 100 + 5
        assert expected_trainables == trainable_params
        assert total_params == trainable_params