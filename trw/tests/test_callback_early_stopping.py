import os
import tempfile
import unittest
import trw
from trw.utils import ExceptionAbortRun
from trw.callbacks import CallbackEarlyStopping
from trw.hparams import HyperParameters, RunResult
from trw.train import Options


class TestCallbackEarlyStopping(unittest.TestCase):
    def test_no_early_stop(self):
        nb_epochs = 100
        nb_runs = 20
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 1.0} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(store=store, loss_fn=lambda hstep: hstep['1-accuracy'])

        # we are better, do NOT raise `ExceptionAbortRun`
        options = Options(num_epochs=nb_epochs)
        early_stopping(options, history[:10], None)
        early_stopping(options, history[:25], None)
        early_stopping(options, history[:50], None)
        early_stopping(options, history[:75], None)

        expected_checkpoints = {
            10: 1.0,
            25: 1.0,
            50: 1.0,
            75: 1.0,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints

    def test_early_stop(self):
        nb_epochs = 100
        nb_runs = 20
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 0.8} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(store=store, loss_fn=lambda hstep: hstep['1-accuracy'])

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = Options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun:
            aborted_run = True

        assert aborted_run, 'run should have been aborted!'

        expected_checkpoints = {
            10: 0.8,
            25: 0.8,
            50: 0.8,
            75: 0.8,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints

    def test_notenough_runs_stop(self):
        nb_epochs = 100
        nb_runs = 2
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 0.8} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(
            store=store,
            loss_fn=lambda hstep: hstep['1-accuracy'],
            only_consider_full_run=False
        )

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = Options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun:
            aborted_run = True

        assert not aborted_run, 'run should NOT have been aborted!'

        expected_checkpoints = {
            10: None,
            25: None,
            50: None,
            75: None,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints

    def test_early_termination(self):
        nb_epochs = 2
        nb_runs = 1
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        for r in range(nb_runs):
            history = [{'1-accuracy': 0.99} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.9999} for n in range(nb_epochs)]

        def check_min_loss(loss, history):
            return loss >= 0.9, f'loss={0.9}'

        early_stopping = CallbackEarlyStopping(
            store=store,
            loss_fn=lambda hstep: hstep['1-accuracy'],
            raise_stop_fn=check_min_loss,
            min_number_of_runs=0
        )

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = Options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun as e:
            aborted_run = True
            assert 'Early termination' in e.reason

        assert aborted_run, 'run should NOT have been aborted!'

    def test_consider_full_run_only(self):
        nb_epochs = 100
        nb_runs = 2
        tmp = tempfile.mkdtemp()

        store_path = os.path.join(tmp, 'store.pkl')
        store = trw.hparams.RunStoreFile(store_path)
        hparams = HyperParameters()

        # partial runs
        for r in range(nb_runs * 10):
            history = [{'1-accuracy': 0.0} for n in range(nb_epochs - 40)]  # last checkpoint is missed!
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        # full runs
        for r in range(nb_runs):
            history = [{'1-accuracy': 0.8} for n in range(nb_epochs)]
            store.save_run(RunResult(metrics={}, hyper_parameters=hparams, history=history, info=None))

        history = [{'1-accuracy': 0.79} for n in range(nb_epochs)]
        early_stopping = CallbackEarlyStopping(
            store=store,
            loss_fn=lambda hstep: hstep['1-accuracy'],
            only_consider_full_run=True,
            min_number_of_runs=0
        )

        # we are better, do NOT raise `ExceptionAbortRun`
        aborted_run = False
        try:
            options = Options(num_epochs=nb_epochs)
            early_stopping(options, history[:10], None)
        except ExceptionAbortRun:
            aborted_run = True

        assert not aborted_run, 'if considering only full results, we have only 2 runs. Run should not be aborted!'

        expected_checkpoints = {
            10: 0.8,
            25: 0.8,
            50: 0.8,
            75: 0.8,
        }
        assert early_stopping.max_loss_by_epoch == expected_checkpoints


if __name__ == '__main__':
    unittest.main()
