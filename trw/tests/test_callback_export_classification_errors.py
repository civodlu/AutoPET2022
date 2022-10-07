import unittest
import trw
import torch.nn as nn
import torch
import os
import numpy as np
from trw.callbacks import CallbackExportClassificationReport


class Criterion:
    def __call__(self, output, truth):
        return torch.zeros(len(output))


def identity(x):
    return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        x = batch['x']
        return {
            'classification_output': trw.train.OutputClassification(
                x,
                batch['x'],
                classes_name='x_truth',
                criterion_fn=Criterion, output_postprocessing=identity)
        }


def create_datasets():
    x = torch.arange(0, 10)
    y = x.clone()
    y[0] = 4
    y[1] = 4

    batch = {
        'x': x,
        'x_truth': y,
        'image_rgb': torch.randn([10, 3, 32, 32], dtype=torch.float32),
        'image_g1': torch.randn([10, 1, 32, 32], dtype=torch.float32),
        'image_g0': torch.randn([10, 32, 32], dtype=torch.float32),
    }
    sampler = trw.train.SamplerSequential(batch_size=10)
    split = trw.train.SequenceArray(batch, sampler=sampler)
    return {
        'dataset1': {
            'split1': split
        }
    }


class TestCallbackExportClassificationErrors(unittest.TestCase):
    def test_classification_report(self):
        output_mappings = {
            'good': {
                'mappinginv': {
                    0: 'str_0',
                    1: 'str_1',
                }
            }
        }

        datasets_infos = {
            'dataset1': {
                'split1': {
                    'output_mappings': output_mappings
                }
            }
        }

        output_raw = np.random.randn(10, 2)
        truth = np.random.randint(0, 2, 10)
        outputs = {
            'dataset1': {
                'split1': {
                    'output1': {
                        'output_ref': trw.train.OutputClassification(torch.zeros([10, 2]), torch.zeros([10, 1]).long(), classes_name='good'),
                        'output_raw': output_raw,
                        'output': np.argmax(output_raw, axis=1),
                        'output_truth': truth
                    }
                }
            }
        }

        callback = CallbackExportClassificationReport()
        options = trw.train.Options(device=torch.device('cpu'))
        options.workflow_options.current_logging_directory = os.path.join(
            options.workflow_options.logging_directory,
            'test_classification_report')
        root_output = options.workflow_options.current_logging_directory
        trw.train.create_or_recreate_folder(options.workflow_options.current_logging_directory)
        callback(options, None, None, None, outputs, None, datasets_infos, None)

        path_report = os.path.join(root_output, 'output1-dataset1-split1-report.txt')
        path_roc = os.path.join(root_output, 'output1-dataset1-split1-ROC.png')
        path_cm = os.path.join(root_output, 'output1-dataset1-split1-cm.png')
        assert os.path.exists(path_report)
        assert os.path.exists(path_roc)
        assert os.path.exists(path_cm)

        with open(path_report, 'r') as f:
            lines = ''.join(f.readlines())

        # make sure the class mapping was correct
        assert 'str_0' in lines
        assert 'str_1' in lines


if __name__ == '__main__':
    unittest.main()
