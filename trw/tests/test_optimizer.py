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
        print(self.layer.weight)
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


class TestOptimizer(unittest.TestCase):
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
        optimizers_fn = partial(trw.train.create_sgd_optimizers_scheduler_one_cycle_lr_fn,
                                max_learning_rate=max_learning_rate,
                                epochs=num_epochs,
                                steps_per_epoch=len(a),
                                learning_rate_start_div_factor=div_factor,
                                learning_rate_end_div_factor=final_div_factor,
                                additional_scheduler_kwargs={'pct_start': fraction_increasing})
        trainer.fit(options, datasets, model, optimizers_fn)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)

        max_index = lrs.argmax()
        assert abs((max_index + 1) / len(lrs) - fraction_increasing) < 1e-8
        assert abs(lrs[max_index] - max_learning_rate) < 1e-8
        assert abs(lrs[0] - max_learning_rate / div_factor) < 1e-8
        assert abs(lrs[-1] - max_learning_rate / div_factor / final_div_factor) < 1e-8
        assert abs(float(model.layer.weight) - 1.0) < 0.1

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_onecycle_sgd_per_epoch(self):
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
        optimizers_fn = partial(trw.train.create_sgd_optimizers_scheduler_one_cycle_lr_fn,
                                max_learning_rate=max_learning_rate,
                                epochs=num_epochs,
                                steps_per_epoch=0,
                                learning_rate_start_div_factor=div_factor,
                                learning_rate_end_div_factor=final_div_factor,
                                additional_scheduler_kwargs={'pct_start': fraction_increasing})
        trainer.fit(options, datasets, model, optimizers_fn)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)

        max_index = lrs.argmax()
        assert abs((max_index + 1) / len(lrs) - fraction_increasing) < 2e-3
        assert abs(lrs[max_index] - max_learning_rate) < 1e-4
        assert abs(lrs[0] - max_learning_rate / div_factor) < 1e-8
        assert abs(lrs[-1] - max_learning_rate / div_factor / final_div_factor) < 1e-4
        assert abs(float(model.layer.weight) - 1.0) < 0.1

        lrs_decreasing = lrs[max_index:]
        step = lrs_decreasing[:-1] - lrs_decreasing[1:]
        assert step.max() > 0
        assert step.min() >= 0

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_onecycle_adam(self):
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
        optimizers_fn = partial(trw.train.create_adam_optimizers_scheduler_one_cycle_lr_fn,
                                max_learning_rate=max_learning_rate,
                                epochs=num_epochs,
                                steps_per_epoch=len(a),
                                learning_rate_start_div_factor=div_factor,
                                learning_rate_end_div_factor=final_div_factor,
                                additional_scheduler_kwargs={'pct_start': fraction_increasing})
        trainer.fit(options, datasets, model, optimizers_fn)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)

        max_index = lrs.argmax()
        assert abs((max_index + 1) / len(lrs) - fraction_increasing) < 1e-8
        assert abs(lrs[max_index] - max_learning_rate) < 1e-8
        assert abs(lrs[0] - max_learning_rate / div_factor) < 1e-8
        assert abs(lrs[-1] - max_learning_rate / div_factor / final_div_factor) < 1e-8
        assert abs(float(model.layer.weight) - 1.0) < 0.1

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_onecycle_adam_per_epoch(self):
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
        optimizers_fn = partial(trw.train.create_adam_optimizers_scheduler_one_cycle_lr_fn,
                                max_learning_rate=max_learning_rate,
                                epochs=num_epochs,
                                steps_per_epoch=len(a),
                                learning_rate_start_div_factor=div_factor,
                                learning_rate_end_div_factor=final_div_factor,
                                additional_scheduler_kwargs={'pct_start': fraction_increasing})
        trainer.fit(options, datasets, model, optimizers_fn)

        lrs = np.asarray(callback_per_step.lr_optimizers['dataset1'])
        assert len(lrs) == num_epochs * len(a)

        max_index = lrs.argmax()
        assert abs((max_index + 1) / len(lrs) - fraction_increasing) < 2e-3
        assert abs(lrs[max_index] - max_learning_rate) < 1e-4
        assert abs(lrs[0] - max_learning_rate / div_factor) < 1e-8
        assert abs(lrs[-1] - max_learning_rate / div_factor / final_div_factor) < 1e-4
        assert abs(float(model.layer.weight) - 1.0) < 0.1

        lrs_decreasing = lrs[max_index:]
        step = lrs_decreasing[:-1] - lrs_decreasing[1:]
        assert step.max() > 0
        assert step.min() >= 0


if __name__ == '__main__':
    unittest.main()
