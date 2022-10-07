import collections
import unittest
from functools import partial

import torch
import trw
import numpy as np
from torch import nn
from trw.callbacks import Callback
from trw.train import Options
from trw.utils import torch_requires


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=False)

    def forward(self, i):
        c = i['input1'].float()
        o = self.layer(c)
        loss = (o - c).abs()
        return {
            'Loss': trw.train.OutputLoss(loss)
        }


class CallbackLearningRateRecorderPerStep(Callback):
    """
    Record the learning rate of the optimizers.

    This is useful as a debugging tool.
    """
    def __init__(self):
        self.lr_optimizers = collections.defaultdict(list)

    # options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs
    def __call__(self, dataset_name, **kwargs):
        optimizer = kwargs.get('optimizer')
        if optimizer is not None:
            lr = optimizer.param_groups[0].get('lr')
            self.lr_optimizers[dataset_name].append(lr)


class CallbackLearningRateRecorderPerEpoch(Callback):
    """
    Record the learning rate of the optimizers.

    This is useful as a debugging tool.
    """
    def __init__(self):
        self.lr_optimizers = collections.defaultdict(list)

    # options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs
    def __call__(self, options, history, model, **kwargs):
        optimizers = kwargs.get('optimizers')
        if optimizers is not None:
            for dataset_name, optimizer in optimizers.items():
                lr = optimizer.param_groups[0].get('lr')
                self.lr_optimizers[dataset_name].append(lr)


class TestOptimizerV2(unittest.TestCase):
    def test_sgd_step_lr(self):
        a = np.arange(0, 10)
        datasets = {
            'dataset1': {
                'train': trw.train.SequenceArray({'input1': a})
            }
        }
        num_epochs = 90

        model = Model()
        options = Options(device=torch.device('cpu'), num_epochs=num_epochs)
        callback_per_step = CallbackLearningRateRecorderPerStep()
        trainer = trw.train.TrainerV2(
            callbacks_per_batch_loss_terms=[callback_per_step],
            callbacks_pre_training=None,
            callbacks_post_training=None,
            callbacks_per_epoch=None
        )

        optimizers = trw.train.OptimizerSGD(learning_rate=0.1).scheduler_step_lr(step_size=30, gamma=0.1)
        trainer.fit(options, datasets, model, optimizers)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)
        assert (abs(lrs[0:10 * 30] - 0.1) < 1e-5).all()
        assert (abs(lrs[10 * 30:2 * 10 * 30] - 0.01) < 1e-5).all()
        assert (abs(lrs[2 * 10 * 30: ] - 0.001) < 1e-5).all()

    def test_option(self):
        options = Options(device=torch.device('cpu'), num_epochs=10)

        options_str = str(options)
        assert 'runtime' in options_str
        assert 'current_logging_directory' in options_str
        assert 'gradient_scaler' in options_str
        assert 'num_epochs' in options_str
        assert 'gradient_update_frequency' in options_str
        assert 'sql_database' in options_str
        assert 'workflow_options' in options_str

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_onecycle_sgd(self):
        a = np.arange(0, 10)
        datasets = {
            'dataset1': {
                'train': trw.train.SequenceArray({'input1': a})
            }
        }
        num_epochs = 100

        model = Model()
        options = Options(device=torch.device('cpu'), num_epochs=num_epochs)
        callback_per_step = CallbackLearningRateRecorderPerStep()
        trainer = trw.train.TrainerV2(
            callbacks_per_batch_loss_terms=[callback_per_step],
            callbacks_pre_training=None,
            callbacks_post_training=None,
            callbacks_per_epoch=None
        )

        max_learning_rate = 0.1
        div_factor = 100
        final_div_factor = 10
        fraction_increasing = 0.2
        optimizers_fn = trw.train.OptimizerSGD(learning_rate=0, momentum=0).scheduler_one_cycle(
            epochs=num_epochs,
            max_learning_rate=max_learning_rate,
            steps_per_epoch=len(a),
            learning_rate_start_div_factor=div_factor,
            learning_rate_end_div_factor=final_div_factor,
            percentage_cycle_increase=fraction_increasing
        )
        trainer.fit(options, datasets, model, optimizers_fn)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)

        max_index = lrs.argmax()
        assert abs((max_index + 1) / len(lrs) - fraction_increasing) < 1e-8
        assert abs(lrs[max_index] - max_learning_rate) < 1e-8
        assert abs(lrs[0] - max_learning_rate / div_factor) < 1e-8
        assert abs(lrs[-1] - max_learning_rate / div_factor / final_div_factor) < 1e-8
        assert abs(float(model.layer.weight) - 1.0) < 0.1


    def test_scheduler_warm_restart(self):
        a = np.arange(0, 10)
        datasets = {
            'dataset1': {
                'train': trw.train.SequenceArray({'input1': a})
            }
        }
        num_epochs = 1000

        model = Model()
        options = Options(device=torch.device('cpu'), num_epochs=num_epochs)
        callback_per_step = CallbackLearningRateRecorderPerEpoch()
        trainer = trw.train.TrainerV2(
            callbacks_per_batch_loss_terms=None,
            callbacks_pre_training=None,
            callbacks_post_training=None,
            callbacks_per_epoch=[callback_per_step]
        )

        optimizers_fn = trw.train.OptimizerSGD(learning_rate=0.5).scheduler_cosine_annealing_warm_restart_decayed(T_0=200, decay_factor=0.7)
        trainer.fit(options, datasets, model, optimizers_fn)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])        

        assert len(lrs) == num_epochs
        assert abs(lrs[199] - 0.5 * 0.7 ** 1) < 1e-5
        assert lrs[198] < 1e-4
        assert abs(lrs[399] - 0.5 * 0.7 ** 2) < 1e-5
        assert lrs[398] < 1e-4
        assert abs(lrs[599] - 0.5 * 0.7 ** 3) < 1e-5
        assert lrs[798] < 1e-4
        assert abs(lrs[799] - 0.5 * 0.7 ** 4) < 1e-5
        assert lrs[998] < 1e-4
        assert abs(lrs[999] - 0.5 * 0.7 ** 5) < 1e-5

if __name__ == '__main__':
    unittest.main()