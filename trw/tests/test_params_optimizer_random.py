import os
from unittest import TestCase

import trw
import trw.hparams
import trw.train
import functools
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import trw.utils

import utils
import numpy as np


def log_nothing(str):
    pass


optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.11)


class Model_XOR(nn.Module):
    """
    XOR problem with hyper-parameter search.

    The only hyper-parameter is the use or not of a non-linear. We know that linear classifier can't classify
    XOR, only the non-linear can. Make sure we can find the expected hyper-parameter
    """
    def __init__(self, hparams):
        super(Model_XOR, self).__init__()
        self.hparams = hparams
        with_dense = hparams.create(trw.hparams.DiscreteBoolean('with_dense', False))

        self.dense1 = nn.Linear(2, 4)
        self.dense2 = None
        if with_dense:
            self.dense2 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, batch):
        x = self.dense1(batch['var_x1'])
        if self.dense2 is not None:
            x = F.relu(self.dense2(x))
        x = self.output(x)
        o = trw.train.OutputRegression(output=x, output_truth=batch['var_y'])
        return {'regression': o}


class TestParamsOptimizer(TestCase):
    def test_discrete_values(self):
        trw.hparams.HyperParameterRepository.reset()
        np.random.seed(0)
        params = trw.hparams.HyperParameters()

        choices = ['choice_1', 'choice_2', 'choice_3']
        params.create(trw.hparams.DiscreteValue('choice', 'choice_1', choices))

        ps = []
        nb_tests = 10000
        for n in range(nb_tests):
            params.randomize()
            p = params.create(trw.hparams.DiscreteValue('choice', 'choice_1', choices))
            assert p in choices
            ps.append(p)

        ps_counts = collections.Counter(ps)
        for choice in choices:
            self.assertTrue(choice in ps_counts)
            self.assertTrue(ps_counts[choice] >= 0.9 * nb_tests / len(choices))

    def test_random_search_single(self):
        #
        # very simple test: minimize x, y given the loss l(x, y) = x^2 + y^2. Best params: x, y = (0, 0)
        #
        trw.hparams.HyperParameterRepository.reset()
        mapping = {
            'value_0': 1000,
            'value_1': 800,
            'value_2': 100,
            'value_3': 0
        }

        def discrete_mapping(value):
            return trw.hparams.DiscreteMapping(
                'param_d',
                value,
                mapping)
        
        def score(x, y, d):
            return x * x + y * y + d

        def evaluate_hparams(hparams):
            x = hparams.create(trw.hparams.ContinuousUniform('param_x', 5, -10, 10))
            y = hparams.create(trw.hparams.ContinuousUniform('param_y', 5, -10, 10))
            d = hparams.create(discrete_mapping('value_0'))
            return {'loss': score(x, y, d)}, [], {'param_x': x, 'param_y': y, 'param_d': d}

        np.random.seed(0)
        optimizer = trw.hparams.HyperParametersOptimizerRandomSearchLocal(
            evaluate_fn=evaluate_hparams,
            log_string=log_nothing,
            repeat=10000)

        options = trw.train.Options(num_epochs=1000)
        store_location = os.path.join(options.workflow_options.logging_directory, 'ut_store.pkl')
        store = trw.hparams.RunStoreFile(store_location=store_location)
        hparams = trw.hparams.HyperParameterRepository.current_hparams
        tries = optimizer.optimize(store, hyper_parameters=hparams)

        # existing hparam must have been replaced by the new one!
        assert trw.hparams.HyperParameterRepository.current_hparams is hparams

        values = [t.metrics['loss'] for t in tries]
        best_try = tries[np.argmin(values)]
        print('BEST=', str(best_try.metrics['loss']))
        self.assertTrue(best_try.metrics['loss'] < 1.0)
        
        # make sure the returned score is the same as if we
        # calculate it directly from the best parameters
        for t in tries[:10]:
            value = t.metrics['loss']
            hparams = t.hyper_parameters
            params = t.info
            assert params['param_x'] == hparams.hparams['param_x'].current_value
            assert params['param_y'] == hparams.hparams['param_y'].current_value
            assert params['param_d'] == hparams.hparams['param_d'].mapping[hparams.hparams['param_d'].current_value]
            assert score(params['param_x'], params['param_y'], params['param_d']) == value

    def test_xor(self):
        #
        # test simple hparam optimization
        #
        def inputs_fn():
            datasets = collections.OrderedDict()
            datasets['dataset_1'] = {
                'train': torch.utils.data.DataLoader(
                    utils.NumpyDatasets(
                        var_x1=np.asarray([[-1, -1], [1, -1], [-1, 1], [1, 1]]),
                        var_y=np.asarray([0, 1, 1, 0])
                    ),
                    batch_size=100
                )
            }
            return datasets

        def model_fn():
            return Model_XOR(trw.hparams.HyperParameterRepository.current_hparams)

        def evaluate_hparams(hparams, options):
            trainer = trw.train.TrainerV2(
                callbacks_per_epoch=None,
                callbacks_pre_training=None,
                callbacks_post_training=None
            )
            output = trainer.fit(
                options,
                datasets=inputs_fn(),
                model=model_fn(),
                optimizers_fn=optimizer_fn,
                log_path=prefix)
            loss = output.outputs['dataset_1']['train']['regression']['loss']
            return {'loss': trw.utils.to_value(loss)}, [], 'no report'

        trw.hparams.HyperParameterRepository.reset()
        np.random.seed(0)
        prefix = 'hparams'
        options = trw.train.Options(num_epochs=1000)
        hparams_copy = trw.hparams.HyperParameterRepository.current_hparams
        optimizer = trw.hparams.params_optimizer_random_search.HyperParametersOptimizerRandomSearchLocal(
            evaluate_fn=functools.partial(evaluate_hparams, options=options),
            log_string=log_nothing,
            repeat=10)

        # we did not provide a new HParams, it should be re-using the current hparams!
        assert hparams_copy is trw.hparams.HyperParameterRepository.current_hparams

        store_location = os.path.join(options.workflow_options.logging_directory, 'ut_store.pkl')
        store = trw.hparams.RunStoreFile(store_location=store_location)
        tries = optimizer.optimize(store)

        values = [t.metrics['loss'] for t in tries]
        best_try = tries[np.argmin(values)]
        print(str(best_try))
        self.assertTrue(len(tries) == 10)
        self.assertTrue(best_try.metrics['loss'] < 1e-5)
