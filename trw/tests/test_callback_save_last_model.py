from unittest import TestCase

import torch
import trw.train
import torch.nn as nn
import tempfile
import glob
import os
import numpy as np
from trw.callbacks.callback_save_last_model import exclude_large_embeddings


class ModelDense(nn.Module):
    def __init__(self, n_output=5):
        super().__init__()

        self.d1 = nn.Linear(10, 100)
        self.d2 = nn.Linear(100, n_output)

    def forward(self, batch):
        return self.d2(self.d1(batch['input']))


class ModelSimple(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, batch):
        l = batch['input'] / self.w
        o = trw.train.OutputEmbedding(l)
        return {'output_1': o}


class TestCallbackSaveLastModel(TestCase):
    def test_rolling_models(self):
        callback = trw.callbacks.CallbackSaveLastModel(
            model_name='checkpoint',
            with_outputs=True,
            is_versioned=True,
            rolling_size=5)

        model = ModelDense()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.Options(logging_directory=logging_directory)
        history = []
        for epoch in range(20):
            history.append({})
            callback(options, history, model, None, {'results': {'outputs': {}}}, None, {'info': 'dummy'}, None)

        models = glob.glob(os.path.join(logging_directory, '*.model'))
        results = glob.glob(os.path.join(logging_directory, '*.model.metadata'))
        assert len(models) == 5
        assert len(results) == 5

        oldest_model = os.path.split(sorted(models)[0])[1]
        assert oldest_model == 'checkpoint_e_16.model'

    def test_keep_best_model(self):
        callback = trw.callbacks.CallbackSaveLastModel(
            keep_model_with_best_metric=trw.callbacks.ModelWithLowestMetric(
                dataset_name='dataset1',
                split_name='split1',
                output_name='output1',
                metric_name='metric1',
                minimum_metric=0.2
            ))

        model = ModelDense()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.Options(logging_directory=logging_directory)

        outputs = {
            'dataset1': {
                'split1': {
                    'output1': {
                        'metric1': 0.9
                    }
                }
            }
        }
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        # metric was not good enough. Do not save!
        models = glob.glob(os.path.join(logging_directory, '*.model'))
        assert len(models) == 1
        assert os.path.basename(models[0]) == 'last.model'

        # update with a better metric: expect `best model` to be exported!
        outputs['dataset1']['split1']['output1']['metric1'] = 0.01
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        models = glob.glob(os.path.join(logging_directory, '*.model'))
        models = sorted(models, reverse=True)
        assert len(models) == 2
        assert os.path.basename(models[-1]) == 'best.model'

        # make sure we don't export worst model
        outputs['dataset1']['split1']['output1']['metric1'] = 0.1
        os.remove(models[-1])
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        models = glob.glob(os.path.join(logging_directory, '*.model'))
        assert len(models) == 1

    def test_post_process_outputs(self):
        callback = trw.callbacks.CallbackSaveLastModel(post_process_outputs=exclude_large_embeddings, with_outputs=True)

        model = ModelDense()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.Options(logging_directory=logging_directory)

        outputs = {
            'dataset1': {
                'split1': {
                    'output1': {
                        'output': np.zeros([1, 20000]),
                        'output_ref': trw.train.outputs_trw.OutputEmbedding(np.zeros([10, 1]))
                    }
                }
            }
        }
        callback(options, [outputs], model, None, outputs, None, {'info': 'dummy'}, None)

        # metric was not good enough. Do not save!
        models = glob.glob(os.path.join(logging_directory, '*.model'))
        assert len(models) == 1
        assert os.path.basename(models[0]) == 'last.model'

        _, results_reloaded = trw.train.TrainerV2.load_model(models[0], with_result=True)
        # the embedding value should be stripped
        assert len(results_reloaded.outputs['dataset1']['split1']['output1']) == 0

    def test_Nan_revert(self):
        model = ModelSimple()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.Options(logging_directory=logging_directory)
        history = []

        callback = trw.callbacks.CallbackSaveLastModel(revert_if_nan_metrics=('NOTPRESENT', 'sub_metric_1a', 'sub_metric_2a'))

        h_step = {
            'dataset_1': {
                'split_1': {
                    'metric_1': {
                        'sub_metric_1a': 1.0
                    },
                    'metric_2': {
                        'sub_metric_2a': 1.0
                    },
                    'metric_3': {
                        # not part of the monitored metrics
                        'sub_metric_3a': float('NaN')
                    }
                }
            }
        }
        history.append(h_step)

        callback(options=options, history=history, model=model, losses=None, outputs=None, datasets=None, datasets_infos=None, callbacks_per_batch=None)
        assert callback.last_model_path is not None

        # model should be reverted
        h_step = {
            'dataset_1': {
                'split_1': {
                    'metric_2': {
                        'sub_metric_2a': float('NaN')
                    },
                    'metric_3': {
                        # not part of the monitored metrics
                        'sub_metric_3a': 2.0
                    }
                }
            }
        }
        history.append(h_step)
        model.w.data[:] = 0
        callback(options=options, history=history, model=model, losses=None, outputs=None, datasets=None,
                 datasets_infos=None, callbacks_per_batch=None)
        assert model.w.data[0] == 1.0

    def test_Nan_cant_revert_without_model_exported(self):
        """
        If there was no previously exported model, make sure we don't raise exception
        and we are in a correct state for the next epochs
        """
        model = ModelSimple()
        logging_directory = tempfile.mkdtemp()
        options = trw.train.Options(logging_directory=logging_directory)
        history = []

        callback = trw.callbacks.CallbackSaveLastModel(revert_if_nan_metrics=('sub_metric_1a',))

        h_step = {
            'dataset_1': {
                'split_1': {
                    'metric_1': {
                        'sub_metric_1a': float('NaN')
                    }
                }
            }
        }
        history.append(h_step)

        # don't revert, no previous model saved
        callback(options=options, history=history, model=model, losses=None, outputs=None, datasets=None,
                 datasets_infos=None, callbacks_per_batch=None)

        assert callback.last_model_path is None, 'no model should have been exported, no model reversion'

        h_step = {
            'dataset_1': {
                'split_1': {
                    'metric_1': {
                        'sub_metric_1a': 1.0
                    }
                }
            }
        }
        history.append(h_step)

        # export has happened
        callback(options=options, history=history, model=model, losses=None, outputs=None, datasets=None,
                 datasets_infos=None, callbacks_per_batch=None)
        assert callback.last_model_path is not None
