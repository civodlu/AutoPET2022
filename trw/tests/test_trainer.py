import os
import pickle
from unittest import TestCase
import collections
import numpy as np
import torch
import torch.utils.data
import trw
import trw.train
import trw.utils
from trw.callbacks import Callback, CallbackEpochSummary
from trw.train import default_pre_training_callbacks

import utils
import functools
import torch.nn as nn


def create_simple_regression():
    """
    Create a simple dataset where output = input_1 * 2.0
    """
    datasets = collections.OrderedDict()
    datasets['simple'] = {
        'train': torch.utils.data.DataLoader(
            utils.NumpyDatasets(
                input_1=np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                output=np.array([[2.0], [4.0], [6.0], [8.0], [10.0]])
            ),
            batch_size=100
        )
    }
    return datasets


def create_random_input():
    """
    Create a simple dataset where output = input_1 * 2.0
    """
    datasets = collections.OrderedDict()
    datasets['simple'] = {
        'train': torch.utils.data.DataLoader(
            utils.NumpyDatasets(
                input_1=np.random.rand(10, 100),
            ),
            batch_size=10,
            shuffle=False
        ),
        'valid': torch.utils.data.DataLoader(
            utils.NumpyDatasets(
                input_1=np.random.rand(10, 100),
            ),
            batch_size=10,
            shuffle=False
        )
    }
    return datasets


class CallbackCollectOutput(Callback):
    """
    Export random samples from our datasets

    Just for sanity check, it is always a good idea to make sure our data is loaded and processed
    as expected.
    """
    def __init__(self):
        self.outputs = None

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        self.outputs = outputs


class ModelSimpleRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, batch):
        x = self.w * batch['input_1']
        o = trw.train.OutputRegression(output=x, output_truth=batch['output'])
        return {'regression': o}


class ModelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1), requires_grad=True)
        self.dropout = nn.Dropout()  # CAREFUL: dropout must be part of the model, else model.valid won't work!

    def forward(self, batch):
        x = batch['input_1'] + 0.0 * self.w
        x = self.dropout(x)

        return {
            'regression': trw.train.OutputRegression(x, batch['input_1'])
        }


class ModelEmbeddedOptimizer(nn.Module):
    """
    Model with an embedded optimizer
    """
    def __init__(self):
        super().__init__()
        self.model = ModelSimpleRegression()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.02)

    def forward(self, batch):
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss_term = outputs['regression'].evaluate_batch(batch, is_training=self.training)

        mean_loss = loss_term['loss'].mean()

        if mean_loss.requires_grad:
            mean_loss.backward()
            self.optimizer.step()

        return {}


def create_model():
    m = ModelSimpleRegression()
    return m


optimizer_fn = functools.partial(trw.train.create_sgd_optimizers_fn, learning_rate=0.11)


def create_trainer(callback_per_epoch=[], callback_per_batch=[], callback_per_batch_loss_terms=[]):
    return trw.train.TrainerV2(
        callbacks_per_epoch=callback_per_epoch,
        callbacks_per_batch_loss_terms=callback_per_batch_loss_terms,
        callbacks_post_training=None,
        callbacks_per_batch=callback_per_batch,
        callbacks_pre_training=default_pre_training_callbacks(with_reporting_server=False)
    )


class CallbackRaiseExceptionAbortRun(Callback):
    def __init__(self, epoch_raised=5):
        self.epoch_raised = epoch_raised

    def __call__(self, options, history, model, **kwargs):
        if len(history) == self.epoch_raised:
            raise trw.utils.ExceptionAbortRun(history=history, reason='callback raised this exception!')


class CallbackRaiseRuntimeException(Callback):
    def __init__(self, epoch_raised=5):
        self.epoch_raised = epoch_raised

    def __call__(self, options, history, model, **kwargs):
        if len(history) == self.epoch_raised:
            raise RuntimeError('callback failed!')


class TestTrainer(TestCase):
    def test_simple_regression(self):
        # the simples API test for model fitting
        options = trw.train.Options(num_epochs=200)
        trainer = create_trainer()
        model = create_model()
        results = trainer.fit(
            options,
            datasets=create_simple_regression(),
            model=model,
            optimizers_fn=optimizer_fn,
            eval_every_X_epoch=2)

        coef_found = trw.utils.to_value(list(model.parameters())[0])
        self.assertAlmostEqual(coef_found, 2.0, delta=1e-3)

    def test_model_mode(self):
        # layers may have different behaviour whether they are in train or eval mode (e.g., dropout, batchnorm)
        # make sure the mode is correctly set for the different phases of train and evaluation
        # in this test, capture the dropout layer applied on an input. In `train` mode we expect random drop
        # of the output while in test mode we should get the identity
        options = trw.train.Options(num_epochs=1)
        callback = CallbackCollectOutput()
        trainer = create_trainer(callback_per_epoch=[callback])

        results = trainer.fit(
            options,
            datasets=create_random_input(),
            model=ModelEmbedding(),
            optimizers_fn=optimizer_fn)
        # do NOT use the results: final iteration is performed in eval mode

        # expected the same
        truth = results.outputs['simple']['valid']['regression']['output_truth']
        found = results.outputs['simple']['valid']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) == 0.0)

        # expected same (result is done in in ``eval`` mode for the training loop to remove
        # dropout and additional training during eval)
        truth = results.outputs['simple']['train']['regression']['output_truth']
        found = results.outputs['simple']['train']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) == 0.0)

        truth = callback.outputs['simple']['valid']['regression']['output_truth']
        found = callback.outputs['simple']['valid']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) == 0.0)

        # this should be done in `train` mode
        truth = callback.outputs['simple']['train']['regression']['output_truth']
        found = callback.outputs['simple']['train']['regression']['output']
        diff = np.abs(truth - found)
        self.assertTrue(np.max(diff) > 0.5)

    def test_batch_and_loss_terms_callbacks(self):
        # make sure the per batch callback is called prior to model.forward call and
        # per batch loss terms is called after model.forward
        all_batches = []
        all_loss_terms = []

        def callback_batch(dataset_name, split_name, batch):
            batch['test_name'] = 'test_value'

        def callback_batch_loss_terms(dataset_name, split_name, batch, loss_terms, **kwargs):
            all_batches.append(batch)
            all_loss_terms.append(loss_terms)

        options = trw.train.Options(num_epochs=1)
        trainer = create_trainer(callback_per_batch=[callback_batch], callback_per_batch_loss_terms=[callback_batch_loss_terms])

        _ = trainer.fit(
            options,
            datasets=create_random_input(),
            model=ModelEmbedding(),
            optimizers_fn=optimizer_fn)

        self.assertTrue(len(all_batches) == 4)
        self.assertTrue(len(all_loss_terms) == 4)

        for batch in all_batches:
            self.assertTrue(batch['test_name'] == 'test_value')

        for loss_terms in all_loss_terms:
            self.assertTrue('regression' in loss_terms)

    def test_serialize_module(self):
        # make sure we can serialize and deserialize module with something else than pickle
        model = ModelSimpleRegression()
        model.w.data[:] = 42

        try:
            import dill
            pickle_module = dill
            extension = 'dill'
        except:
            pickle_module = pickle
            extension = 'pickle'

        path = os.path.join(utils.root_output, f'test.{extension}')
        trw.train.TrainerV2.save_model(model, None, path, pickle_module=pickle_module)

        model2, results = trw.train.TrainerV2.load_model(path, pickle_module=pickle_module)
        assert (model2.w == model.w).all()

    def test_embedded_optimizer(self):
        options = trw.train.Options(num_epochs=200)
        trainer = create_trainer()
        model = ModelEmbeddedOptimizer()
        _ = trainer.fit(
            options,
            datasets=create_simple_regression(),
            model=model,
            optimizers_fn=None,  # no optimizer: it is embedded in the model!
            eval_every_X_epoch=2)

        coef_found = trw.utils.to_value(list(model.model.parameters())[0])
        print(coef_found)
        self.assertAlmostEqual(coef_found, 2.0, delta=1e-3)

    def test_run_interrupted(self):
        options = trw.train.Options(num_epochs=50)

        trainer = trw.train.TrainerV2(
            callbacks_per_epoch=[CallbackRaiseExceptionAbortRun(epoch_raised=5)],
        )

        try:
            _ = trainer.fit(
                options,
                datasets=create_simple_regression(),
                model=ModelEmbeddedOptimizer(),
                optimizers_fn=None,  # no optimizer: it is embedded in the model!
                eval_every_X_epoch=2)
            assert 0, 'should have raised exception!'
        except trw.utils.ExceptionAbortRun as e:
            assert len(e.history) == 5

    def test_run_callback_exceptions(self):
        """
        Exception in callbacks should not stop the run
        """
        options = trw.train.Options(num_epochs=50)

        trainer = trw.train.TrainerV2(
            callbacks_per_epoch=[CallbackEpochSummary(), CallbackRaiseRuntimeException()],
            callbacks_post_training=[CallbackRaiseRuntimeException()],
            callbacks_pre_training=[CallbackRaiseRuntimeException()]
        )

        r = trainer.fit(
            options,
            datasets=create_simple_regression(),
            model=ModelEmbeddedOptimizer(),
            optimizers_fn=None,  # no optimizer: it is embedded in the model!
            eval_every_X_epoch=2)

        assert len(r.history) == 50 + 1
        assert len(r.outputs) >= 1