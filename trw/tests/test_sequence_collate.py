from unittest import TestCase

import trw
import trw.train
import numpy as np
import torch
import trw.utils


def make_list_dicts(batch):
    samples = []
    for n in range(10):
        sub_batch = {
            'sample_uid': batch['sample_uid'],
            'volume': torch.zeros([1, 42])
        }
        samples.append(sub_batch)

    return samples


class TestSequenceCollate(TestCase):
    def test_collate_list_dicts(self):
        """
        Test that we can easily combine SequenceArray -> SequenceAsyncReservoir -> SequenceCollate
        """
        nb_indices = 20
        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=10,
                                                    function_to_run=make_list_dicts, min_reservoir_samples=10).collate()

        for batch in sequence:
            assert trw.utils.len_batch(batch) == 10
