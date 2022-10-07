from unittest import TestCase
import trw.train
import numpy as np
import collections
import torch
import trw.train.collate


class TestSequenceBatcher(TestCase):
    def test_split_batcher(self):
        split = {
            'classes': np.asarray([0, 1, 2, 3, 4, 5]),
            'indices': np.asarray([0, 1, 2, 3, 4, 5]),
        }

        indices = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).batch(2)
        for batch in split_np:
            batch = trw.train.collate.default_collate_fn(batch, device=None)
            assert isinstance(batch, collections.Mapping)
            assert len(batch['classes']) == 2
            assert len(batch['indices']) == 2
            indices.append(batch['indices'])

        assert len(indices) == 3
        assert torch.sum(torch.stack(indices)) == 1 + 2 + 3 + 4 + 5

    def test_split_batcher_batch_not_full(self):
        split = {
            'classes': np.asarray([0, 1, 2, 3, 4]),
            'indices': np.asarray([0, 1, 2, 3, 4]),
        }

        indices = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).batch(2)
        for batch in split_np:
            batch = trw.train.collate.default_collate_fn(batch, device=None)
            indices.append(batch['indices'])

        assert len(indices) == 3
        assert len(indices[-1]) == 1

    def test_split_batcher_batch_not_full_discarded(self):
        split = {
            'classes': np.asarray([0, 1, 2, 3, 4]),
            'indices': np.asarray([0, 1, 2, 3, 4]),
        }

        indices = []
        split_np = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).batch(2, discard_batch_not_full=True)
        for batch in split_np:
            batch = trw.train.collate.default_collate_fn(batch, device=None)
            indices.append(batch['indices'])

        assert len(indices) == 2
