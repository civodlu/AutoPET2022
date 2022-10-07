import unittest
import trw
import numpy as np
from torch import nn
from trw.layers.layer_config import PoolType, NormType


class DummyModel(nn.Module):
    def __init__(self):
        # we need to define at least one variable to be optimized!
        super().__init__()
        self.linear = nn.Linear(1, 1)


class TestParamCreator(unittest.TestCase):
    def test_optimizers(self):
        # create a new session
        trw.hparams.HyperParameterRepository.reset()
        a = np.arange(0, 10)
        datasets = {
            'dataset1': {
                'train': trw.train.SequenceArray({'input1': a})
            }
        }
        model = DummyModel()

        for n in range(500):
            o = trw.hparams.create_optimizers_fn(datasets, model, name_prefix='PREFIX_')
            trw.hparams.HyperParameterRepository.current_hparams.randomize()

        hparams = trw.hparams.HyperParameterRepository.current_hparams.hparams
        assert len(hparams) >= 10

        for name, h in hparams.items():
            # all hparams must be prefixed!
            assert 'PREFIX_' in name

    def test_activation(self):
        trw.hparams.HyperParameterRepository.reset()
        v = trw.hparams.create_activation('hp1', default_value=nn.ReLU)
        assert isinstance(v(), nn.Module)

    def test_pool_type(self):
        trw.hparams.HyperParameterRepository.reset()
        v = trw.hparams.create_pool_type('hp1', default_value=PoolType.MaxPool)
        assert isinstance(v, PoolType)

    def test_norm_type(self):
        trw.hparams.HyperParameterRepository.reset()
        v = trw.hparams.create_norm_type('hp1', default_value=NormType.BatchNorm)
        assert isinstance(v, NormType)


if __name__ == '__main__':
    unittest.main()
