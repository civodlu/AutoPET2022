from unittest import TestCase
import trw
import torch.nn as nn
import functools
import glob
import os


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = trw.layers.ConvsBase(2, input_channels=3, channels=[4, 8], strides=[2, 2], with_flatten=True)
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


optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.01)


def callbacks_post_training_fn():
    return [
        trw.callbacks.CallbackExplainDecision()
    ]


class TestCallbackExplainDecision(TestCase):
    def test_synthetic(self):
        """
        Make sure the explanations for the different algorithms are generated. This is not assessing
        The validity of the explanation.
        """
        options = trw.train.Options(num_epochs=50)
        trainer = trw.train.TrainerV2(
            callbacks_post_training=callbacks_post_training_fn(),
            callbacks_pre_training=None)

        results = trainer.fit(
            options,
            datasets=create_dataset(),
            log_path='synthetic_explanation',
            model=Net(),
            optimizers_fn=optimizer_fn)

        classification_error = results.history[-1]['fake_symbols_2d']['train']['fake_symbols_2d']['classification error']
        assert classification_error < 0.05

        expected_algorithms = [kvp.name for kvp in list(trw.callbacks.ExplainableAlgorithm)]
        explanation_path = os.path.join(options.workflow_options.current_logging_directory, 'explained')
        for algorithm in expected_algorithms:
            files = glob.glob(os.path.join(explanation_path, f'*{algorithm}*.png'))
            assert len(files) >= 2 * 4


