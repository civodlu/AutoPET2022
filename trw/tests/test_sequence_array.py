from unittest import TestCase

import trw
import trw.train
import numpy as np
import torch
import trw.train.collate
import trw.utils


def double_values(item):
    item['values'] = item['values'] * 2.0
    return item


class TestSequenceArray(TestCase):
    @staticmethod
    def check_types_and_shapes(use_advanced_indexing):
        # Just make sure that the expected type have been converted to torch.Tensor
        split = {
            'uid': ['test'] * 10,
            'np': np.ones([10, 2]),
            'torch': torch.ones([10, 2]),
            'list': list(range(10)),
        }

        sampler = trw.train.SamplerSequential(batch_size=2)
        sequence = trw.train.SequenceArray(split, sampler=sampler, use_advanced_indexing=use_advanced_indexing).collate()

        batches = []
        for batch in sequence:
            assert len(batch) == 5
            batches.append(batch)

            assert isinstance(batch['uid'], list)  # list of strings: don't convert!
            assert isinstance(batch['np'], torch.Tensor)
            assert isinstance(batch['torch'], torch.Tensor)
            assert isinstance(batch['list'], torch.Tensor)

            assert len(batch['uid']) == 2
            assert len(batch['np']) == 2
            assert len(batch['torch']) == 2
            assert len(batch['list']) == 2

        assert len(batches) == 5

    def test_types_shapes(self):
        TestSequenceArray.check_types_and_shapes(use_advanced_indexing=False)
        TestSequenceArray.check_types_and_shapes(use_advanced_indexing=True)

    @staticmethod
    def check_with_batching(use_advanced_indexing):
        split = {
            'np': np.ones([20, 2]),
            'torch': torch.ones([20, 2]),
            'list': list(range(20)),
            'uid': ['test'] * 20,
        }

        sampler = trw.train.SamplerSequential(batch_size=2)
        sequence = trw.train.SequenceArray(split, sampler=sampler, use_advanced_indexing=use_advanced_indexing).batch(2).collate()

        batches = []
        for batch in sequence:
            assert len(batch) == 5
            batches.append(batch)

            assert isinstance(batch['np'], torch.Tensor)
            assert isinstance(batch['torch'], torch.Tensor)
            assert isinstance(batch['list'], torch.Tensor)

            assert len(batch['np']) == 4
            assert len(batch['torch']) == 4
            assert len(batch['list']) == 4

        assert len(batches) == 5

    def test_with_batching(self):
        #TestSequenceArray.check_with_batching(use_advanced_indexing=False)  # TODO fix this! we want uniform behaviour
        TestSequenceArray.check_with_batching(use_advanced_indexing=True)

    def test_sequence_subsample_uids(self):
        """
        Make sure we can subsample our data by UIDs
        """
        split = {
            'np': np.ones([20, 2]),
            'torch': torch.ones([20, 2]),
            'list': list(range(20)),
            'uids': list(range(20)),
        }

        uids = [10, 8, 4, 99]
        sequence = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        sequence_subsampled = sequence.subsample_uids(uids, uids_name='uids', new_sampler=None)

        for batch_index, batch in enumerate(sequence_subsampled):
            # make sure we kept the ordering of the UIDs
            assert trw.utils.len_batch(batch) == 1
            expected_uids = uids[batch_index]
            assert expected_uids == batch['uids'][0]
            assert expected_uids == batch['list'][0]

        assert batch_index == 2

    def test_sequence_subsample(self):
        """
        Make sure we can subsample our data
        """
        split = {
            'np': np.ones([20, 2]),
            'torch': torch.ones([20, 2]),
            'list': list(range(20)),
            'uids': list(range(20)),
        }

        sequence = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        sequence_subsampled = sequence.subsample(4)

        uids = []
        for batch_index, batch in enumerate(sequence_subsampled):
            assert trw.utils.len_batch(batch) == 1
            assert batch['uids'][0] == batch['list'][0]
            uids.append(batch['uids'][0])

        assert len(set(uids)) == 4

    def test_split_subsample(self):
        nb_indices = 40
        split = {
            'uid': np.asarray(list(range(nb_indices))),
            'values': np.asarray(list(range(nb_indices))),
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(double_values, nb_workers=1).batch(5).subsample(10)

        batches = []
        for i in split:
            i = trw.train.collate.default_collate_fn(i, device=None)
            for n in range(trw.utils.len_batch(i)):
                self.assertTrue(i['uid'].data.numpy()[n] * 2.0 == i['values'][n])
                self.assertTrue(np.max(i['uid'].data.numpy()) < 40)
            batches.append(i)
            print(i)

        # we resampled the sequence to 10 samples with a batch size of 5
        # so we expect 2 batches
        self.assertTrue(len(batches) == 2)
