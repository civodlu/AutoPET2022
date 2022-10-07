from unittest import TestCase
import trw
import numpy as np
import torch
import collections
import trw.train.collate

import trw.utils


class TestCollate(TestCase):
    def test_collate_dicts(self):
        batch = {
            'list': [1, 2, 3],
            'value': 42.0,
            'np': np.ones(shape=[3, 2]),
            'torch': torch.ones([3, 2]),
            'strs': ['1', '2', '3'],
        }

        collated = trw.train.collate.default_collate_fn(batch, device=None)
        self.assertTrue(isinstance(collated, collections.OrderedDict))
        self.assertTrue(isinstance(collated['list'], torch.Tensor))
        self.assertTrue(len(collated['list']) == 3)
        self.assertTrue(isinstance(collated['np'], torch.Tensor))
        self.assertTrue(collated['np'].shape == (3, 2))
        self.assertTrue(isinstance(collated['torch'], torch.Tensor))
        self.assertTrue(collated['torch'].shape == (3, 2))
        self.assertTrue(isinstance(collated['value'], float))
        self.assertTrue(isinstance(collated['strs'], list))

    def test_collate_list_of_dicts(self):
        batch_1 = {
            'np': np.ones(shape=[1, 3, 2]),
            'list': [1],
            'value': 42.0,
            'torch': torch.ones([1, 3, 2]),
            'strs': ['1'],
        }

        batch_2 = {
            'np': np.ones(shape=[1, 3, 2]),
            'list': [2],
            'value': 42.0,
            'torch': torch.ones([1, 3, 2]),
            'strs': ['2'],
        }

        collated = trw.train.collate.default_collate_fn([batch_1, batch_2], device=None)

        self.assertTrue(isinstance(collated, collections.OrderedDict))
        self.assertTrue(trw.utils.len_batch(collated) == 2)
        self.assertTrue(isinstance(collated['list'], torch.Tensor))
        self.assertTrue(len(collated['list']) == 2)
        self.assertTrue(isinstance(collated['np'], torch.Tensor))
        self.assertTrue(collated['np'].shape == (2, 3, 2))
        self.assertTrue(isinstance(collated['torch'], torch.Tensor))
        self.assertTrue(collated['torch'].shape == (2, 3, 2))
        self.assertTrue(isinstance(collated['value'], torch.Tensor))
        self.assertTrue(len(collated['torch']) == 2)
        self.assertTrue(isinstance(collated['strs'], list))
        self.assertTrue(len(collated['strs']) == 2)
        self.assertTrue(isinstance(collated['value'], torch.Tensor))
        self.assertTrue(len(collated['value']) == 2)
