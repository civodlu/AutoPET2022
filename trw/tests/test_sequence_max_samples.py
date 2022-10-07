import unittest

import trw
import numpy as np


class TestSequenceMaxSamples(unittest.TestCase):
    def test_smaller_sub_sequence(self):
        """
        The sequence is cut in smaller sub-sequences. Test
        that the original sequence is NOT restarted.
        """
        nb_samples = 6
        sampler = trw.train.SamplerSequential(batch_size=1)
        sequence = trw.train.SequenceArray({'id': np.arange(nb_samples)}, sampler=sampler)
        sequence = sequence.max_samples(3)

        batches = []
        for b in sequence:
            batches.append(b)

        assert len(batches) == 3
        assert batches[0]['id'][0] == 0
        assert batches[1]['id'][0] == 1
        assert batches[2]['id'][0] == 2

        batches = []
        for b in sequence:
            batches.append(b)

        assert len(batches) == 3
        assert batches[0]['id'][0] == 3
        assert batches[1]['id'][0] == 4
        assert batches[2]['id'][0] == 5

        # here we should have restarted the underlying sequence
        batches = []
        for b in sequence:
            batches.append(b)

        assert len(batches) == 3
        assert batches[0]['id'][0] == 0
        assert batches[1]['id'][0] == 1
        assert batches[2]['id'][0] == 2

    def test_larger_sub_sequence(self):
        """
        Here the number of samples is larger than the underlying sequence.
        There will be an interim sequence restart of the underlying sequence
        to obtain enough samples
        """
        nb_samples = 3
        sampler = trw.train.SamplerSequential(batch_size=1)
        sequence = trw.train.SequenceArray({'id': np.arange(nb_samples)}, sampler=sampler)
        sequence = sequence.max_samples(5)

        batches = []
        for b in sequence:
            batches.append(b)

        assert len(batches) == 5
        for i in range(5):
            assert batches[i]['id'][0] == i % 3

        # use the last element and restart the sequence
        batches = []
        for b in sequence:
            batches.append(b)

        assert len(batches) == 5
        for i in range(5):
            assert batches[i]['id'][0] == (i + 2) % 3


if __name__ == '__main__':
    unittest.main()
