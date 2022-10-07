import trw
import numpy as np
import torch
from unittest import TestCase
import collections
import trw.train.collate

import trw.utils


def make_sequence(size, batch_size):
    d = collections.OrderedDict()
    d['string_list'] = [f'string_{v}' for v in range(size)]
    d['np_array'] = np.zeros([size, 2, 3])
    d['torch_array'] = torch.arange(size)

    sampler = trw.train.SamplerSequential(batch_size=batch_size)
    return trw.train.SequenceArray(d, sampler=sampler)


class TestSequenceSubBatch(TestCase):
    def test_simple(self):
        """
        Split the sequence into smaller batch than original.
        This sequence contains only full batches.
        """
        sequence_source = make_sequence(30, 10)
        sequence = sequence_source.sub_batch(5)

        batches = []
        for batch in sequence:
            batches.append(batch)
            assert trw.utils.len_batch(batch) == 5
        assert len(batches) == (30 // 10) * (10 // 5)

        all_batches = trw.train.collate.default_collate_fn(batches, device=None)
        assert sequence_source.split['string_list'] == all_batches['string_list']

    def test_not_full_batches_not_discarded(self):
        """
        Generate a sequence with batches not full. Make sure no data is discarded.
        """
        sequence_source = make_sequence(30, 15)
        sequence = sequence_source.sub_batch(10)

        batches = []
        for batch in sequence:
            batches.append(batch)
        assert len(batches) == 4

        assert trw.utils.len_batch(batches[0]) == 10
        assert trw.utils.len_batch(batches[1]) == 5
        assert trw.utils.len_batch(batches[2]) == 10
        assert trw.utils.len_batch(batches[3]) == 5

        all_batches = trw.train.collate.default_collate_fn(batches, device=None)
        assert sequence_source.split['string_list'] == all_batches['string_list']

    def test_not_full_batches_discarded(self):
        """
        Generate a sequence with batches not full. Make sure data is discarded.
        """
        sequence_source = make_sequence(30, 15)
        sequence = sequence_source.sub_batch(10, discard_batch_not_full=True)

        batches = []
        for batch in sequence:
            batches.append(batch)
        assert len(batches) == 2
        assert trw.utils.len_batch(batches[0]) == 10
        assert trw.utils.len_batch(batches[1]) == 10

        all_batches = trw.train.collate.default_collate_fn(batches, device=None)
        assert all_batches['torch_array'].numpy().tolist() == np.arange(0, 10).tolist() + np.arange(15, 25).tolist()
