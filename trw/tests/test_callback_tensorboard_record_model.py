import unittest
import trw
import torch.nn as nn
import torch
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = trw.layers.ConvsBase(
            2,
            input_channels=3,
            channels=[4, 8],
            strides=[2, 2],
            with_flatten=True)
        self.classifier = nn.Linear(32, 2)

    def forward(self, batch):
        x = batch['image']
        x = self.convs(x)
        x = self.classifier(x)

        return {
            'fake_symbols_2d': trw.train.OutputClassification(
                x, batch['classification'], classes_name='classification')
        }


def create_dataset():
    return trw.datasets.create_fake_symbols_2d_dataset(
        nb_samples=20,
        global_scale_factor=0.5,
        image_shape=[32, 32],
        nb_classes_at_once=1,
        max_classes=2)


class TestCallbackTensorboardRecordModel(unittest.TestCase):
    def test_basic(self):
        options = trw.train.Options(device=torch.device('cpu'))
        callback = trw.callbacks.CallbackTensorboardRecordModel(onnx_folder='onnx_export')

        onnx_root = os.path.join(options.workflow_options.current_logging_directory, 'onnx_export')
        trw.train.create_or_recreate_folder(onnx_root)

        model = Net()
        datasets = create_dataset()
        callback(options, None, model, None, None, datasets, None, None)
        assert os.path.exists(os.path.join(onnx_root, 'model.onnx'))


if __name__ == '__main__':
    unittest.main()
