from unittest import TestCase
import trw.train
import numpy as np
import time
import torch


def make_volume_torch(batch):
    print('JOB starte', batch['path'])
    batch['volume'] = torch.zeros([1, 91, 110, 91], dtype=torch.float)
    time.sleep(2)
    return batch


class TestSequenceReservoirPerformance(TestCase):
    def test_reservoir_performance(self):
        # basic test to measure performance: we expect the sampling time to be negligible
        # and this whatever time it takes for the worker job
        nb_indices = 800
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 5

        sampler = trw.train.SamplerRandom()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=make_volume_torch,
                                                    min_reservoir_samples=max_reservoir_samples).collate().batch(40)

        print('Iter start')
        time_start = time.time()
        iter(sequence)
        time_end = time.time()
        print('Iter end', time_end - time_start)

        time_ref = (time_end - time_start) / max_reservoir_samples
        print('ref=', time_ref)

        nb_epochs = 10
        for i in range(1, nb_epochs):
            time_start = time.time()
            batches = []
            for batch in sequence:
                batches.append(batch)
            time_end = time.time()
            time_epoch = time_end - time_start
            print('TIME=', time_end - time_start)
            self.assertTrue(time_epoch * 10 < time_ref)
