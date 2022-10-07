from unittest import TestCase

import trw
import trw.train
import numpy as np
import time
import collections
import torch
import functools

import trw.utils


def function_to_run(batch):
    return batch


def function_to_run_id(batch):
    return batch

def function_to_run_id_wait(batch, wait_time: float, text=None):
    time.sleep(wait_time)
    if text is not None:
        print(text, batch)
    return batch

def function_to_run_multiple_wait(batch, nb: int, wait_time: float, text=None):
    time.sleep(wait_time)
    #if text is not None:
    #    print(text, batch)
    v = float(batch['path'])
    return {'path2': torch.arange(v * 100, v * 100 + nb)}


def function_to_run2(batch):
    print('JOB starte', batch['path'])
    time.sleep(0.1)
    return batch

def worker_with_error(batch):
    if batch['sample_uid'][0] == 10:
        raise IndexError('This is an expected exception to test worker recovery from failure!')
    return batch


def make_volume_torch(batch):
    print('JOB starte', batch['path'])
    batch['volume'] = torch.zeros([1, 10, 11, 12], dtype=torch.float)
    time.sleep(0.1)
    return batch


def make_list_dicts(batch, wait_time=None):
    samples = []
    print('sample_uid=', batch['sample_uid'])
    for n in range(10):
        sub_batch = {
            'sample_uid': batch['sample_uid'],
            'volume': torch.zeros([1, 42])
        }
        samples.append(sub_batch)

    if wait_time is not None:
        time.sleep(wait_time)
    return samples


class TestSequenceReservoir(TestCase):
    def test_reservoir_basics2(self):
        # Test the sequence satisfies statistical properties:
        # - items from a sequence must be in equal proportion (they are randomly sampled)
        nb_indices = 10
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 5

        sampler = trw.train.SamplerRandom()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=function_to_run,
                                                    min_reservoir_samples=max_reservoir_samples).collate()

        time.sleep(2)

        time_start = time.time()
        samples = collections.defaultdict(lambda: 0)
        nb_epochs = 100000
        for i in range(1, nb_epochs):
            batches = []
            for batch in sequence:
                nb_samples = trw.utils.len_batch(batch)
                # we requested a single sample at a time
                self.assertTrue(nb_samples == 1)

                p = trw.utils.to_value(batch['path'])
                self.assertTrue(p.shape == (1, 2))
                batches.append(batch)

                value = int(trw.utils.to_value(batch['sample_uid'])[0])
                samples[value] += 1
            self.assertTrue(len(batches) <= max_reservoir_samples)
        time_end = time.time()
        print('TIME=', time_end - time_start)

        expected_counts = nb_epochs / nb_indices * max_reservoir_samples
        for c, counts in samples.items():
            error_percent = abs(counts - expected_counts) / expected_counts
            print(f'c={c}, counts={counts}, expected_counts={expected_counts}, error={error_percent}')
            self.assertTrue(error_percent < 0.1)

    def test_subsample_uid(self):
        """
        Make sure we can resample the sequence with UID
        """
        nb_indices = 800
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 200

        sampler = trw.train.SamplerSequential()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=function_to_run,
                                                    min_reservoir_samples=max_reservoir_samples).collate().batch(40)
        
        subsampled_sequence = sequence.subsample_uids(uids=np.arange(200, 300), uids_name=trw.train.default_sample_uid_name)
        
        nb_samples = 0
        values = set()
        for batch in subsampled_sequence:
            batch_set = set(trw.utils.to_value(batch['sample_uid']))
            nb_samples += trw.utils.len_batch(batch)
            values = values.union(batch_set)

        assert len(values) == 100
        assert np.min(list(values)) == 200
        assert np.max(list(values)) == 299

    def test_reservoir_batch(self):
        """
        Test that we can easily combine SequenceArray -> SequenceAsyncReservoir -> SequenceBatch
        """
        nb_indices = 20
        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=10,
                                                    function_to_run=make_list_dicts, min_reservoir_samples=10).batch(5)

        for batch in sequence:
            assert trw.utils.len_batch(batch) == 5 * 10, 'found={}, expected={}'.format(
                trw.utils.len_batch(batch), 5 * 10)

    def test_fill_reservoir_every_epoch(self):
        """
        The reservoir will start tasks and retrieve results every epoch
        """
        max_jobs_at_once = 5
        nb_indices = 30
        nb_epochs = 7
        min_reservoir_samples = 5
        max_reservoir_samples = 20

        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=functools.partial(make_list_dicts, wait_time=0.02),
                                                    min_reservoir_samples=min_reservoir_samples,
                                                    max_jobs_at_once=max_jobs_at_once)

        for epoch in range(nb_epochs):
            print('sleeping')
            time.sleep(0.5)
            print('epoch=', epoch)
            expected_reservoir_size = min(epoch * max_jobs_at_once, max_reservoir_samples)
            assert sequence.reservoir_size() >= expected_reservoir_size, 'found={}, expected={}'.format(sequence.reservoir_size(), expected_reservoir_size)
            assert sequence.reservoir_size() <= expected_reservoir_size + max_jobs_at_once, 'found={}, expected={}'.format(sequence.reservoir_size(), expected_reservoir_size)
            print('found={}, expected={}'.format(sequence.reservoir_size(), expected_reservoir_size))
            for batch_id, batch in enumerate(sequence):
                if batch_id == 0 and epoch == 0:
                    assert sequence.reservoir_size() >= min_reservoir_samples and \
                           sequence.reservoir_size() <= min_reservoir_samples + max_jobs_at_once

    def test_worker_error(self):
        nb_indices = 20

        split = {'path': np.asarray(np.arange(nb_indices))}
        sampler = trw.train.SamplerSequential(batch_size=1)
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)

        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=10,
                                                    function_to_run=worker_with_error, min_reservoir_samples=5,
                                                    max_jobs_at_once=1).collate()

        for n in range(100):
            batches = []
            for batch in sequence:
                assert batch['sample_uid'][0] != 10, 'this job should have failed!'
                batches.append(batch)
            assert len(batches) >= 5

    def test_multiple_iterators_same_sequence(self):
        # test the statistics of iterating the same sequence using different iterators
        np.random.seed(0)
        torch.random.manual_seed(0)

        nb_indices = 10
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 5

        sampler = trw.train.SamplerRandom()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=function_to_run,
                                                    min_reservoir_samples=max_reservoir_samples,
                                                    reservoir_sampler=trw.train.SamplerRandom()).collate()

        samples_0 = collections.defaultdict(lambda: 0)
        samples_1 = collections.defaultdict(lambda: 0)
        nb_epochs = 100000
        for i in range(1, nb_epochs):
            it_0 = iter(sequence)
            it_1 = iter(sequence)
            for batch in it_0:
                nb_samples = trw.utils.len_batch(batch)
                self.assertTrue(nb_samples == 1)
                value = int(trw.utils.to_value(batch['sample_uid'])[0])
                samples_0[value] += 1

            for batch in it_1:
                nb_samples = trw.utils.len_batch(batch)
                self.assertTrue(nb_samples == 1)
                value = int(trw.utils.to_value(batch['sample_uid'])[0])
                samples_1[value] += 1

        expected_counts = nb_epochs / nb_indices * max_reservoir_samples
        for c, counts in samples_0.items():
            error_percent = abs(counts - expected_counts) / expected_counts
            print(f'c={c}, counts={counts}, expected_counts={expected_counts}, error={error_percent}')
            self.assertTrue(error_percent < 0.1)

        for c, counts in samples_1.items():
            error_percent = abs(counts - expected_counts) / expected_counts
            print(f'c={c}, counts={counts}, expected_counts={expected_counts}, error={error_percent}')
            self.assertTrue(error_percent < 0.1)

    def test_subsample(self):
        nb_indices = 10
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 5

        sampler = trw.train.SamplerRandom()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=function_to_run,
                                                    min_reservoir_samples=max_reservoir_samples,
                                                    reservoir_sampler=trw.train.SamplerRandom()).collate().subsample(2)

        uids = set()
        for epoch in range(10):
            for batch in sequence:
                value = int(trw.utils.to_value(batch['sample_uid'])[0])
                uids.add(value)
        assert len(uids) == 2

    def test_reservoir_maximum_replacement(self):
        """
        Make sure we can control at which rate the content of the reservoir is replaced per epoch
        """
        nb_indices = 100
        paths = [[i, 42] for i in range(nb_indices)]
        split = {'path': np.asarray(paths)}
        max_reservoir_samples = 20
        max_reservoir_replacement_size = 5
        max_jobs_at_once = 100

        sampler = trw.train.SamplerSequential()
        numpy_sequence = trw.train.SequenceArray(split, sampler=sampler)
        sequence = trw.train.SequenceAsyncReservoir(numpy_sequence, max_reservoir_samples=max_reservoir_samples,
                                                    function_to_run=function_to_run_id,
                                                    min_reservoir_samples=max_reservoir_samples,
                                                    max_jobs_at_once=max_jobs_at_once,
                                                    max_reservoir_replacement_size=max_reservoir_replacement_size,
                                                    reservoir_sampler=trw.train.SamplerSequential())

        last_uids = set()
        for epoch in range(10):
            print(f'-----epoch={epoch}')
            time.sleep(0.1)
            current_uids = set()
            for batch in sequence:
                assert len(batch) == 1
                uid = batch[0]['sample_uid'][0]
                current_uids.add(uid)

            print(current_uids)
            if epoch > 0:
                d = last_uids.difference(current_uids)
                print(d)
                assert len(d) == max_reservoir_replacement_size, f'found={len(d)}, expected={max_reservoir_replacement_size}'

            last_uids = current_uids

    def test_slow_sequence_reservoir_fast_map_id(self):
        """
        Test the loading of the reservoir doesn't affect the iteration
        of the reservoir (i.e., once the minimum of jobs is loaded,
        iterating the reservoir should be instantaneous)
        """
        nb_indices = 15
        nb_epochs = 4
        split = {'path': np.asarray([[i] for i in range(nb_indices)])}
        max_reservoir_samples = 5
        max_reservoir_replacement_size = 5
        max_jobs_at_once = max_reservoir_samples
        wait_time_reservoir = 1.0
        wait_time_map = 0.5
        numpy_sequence = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            function_to_run=functools.partial(function_to_run_id_wait, wait_time=wait_time_reservoir, text='reservoir_loaded'),
            max_reservoir_samples=max_reservoir_samples,
            min_reservoir_samples=max_reservoir_samples,
            max_jobs_at_once=max_jobs_at_once,
            max_reservoir_replacement_size=max_reservoir_replacement_size,
            reservoir_sampler=trw.train.SamplerSequential())        
        sequence = sequence.map(functools.partial(function_to_run_id_wait, wait_time=wait_time_map), nb_workers=1)

        expected_time = max_reservoir_samples * wait_time_map * (nb_epochs - 1)
        time_start = None
        for epoch in range(nb_epochs):
            if epoch == 1:
                # discard first epoch timing due to processes/threads creation time
                time_start = time.perf_counter()

            epoch_start = time.perf_counter()
            for batch_id, batch in enumerate(sequence):
                print(batch_id, str(batch))
            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            print(f'epoch={epoch}, epoch_time={epoch_time}')
        time_end = time.perf_counter()
        time_taken = time_end - time_start
        print(f'DONE, time_taken={time_taken}, expected_time={expected_time}')
        assert abs(time_taken - expected_time) < 0.5

    def test_slow_sequence_reservoir_fast_map_multiple(self):
        """
        Test the loading of the reservoir doesn't affect the iteration
        of the reservoir (i.e., once the minimum of jobs is loaded,
        iterating the reservoir should be instantaneous).

        Many more batches to process than reservoir size. Since
        there is almost no data transferred between processes
        the overhead SHOULD be minimal.
        """
        nb_indices = 15
        nb_epochs = 3
        split = {'path': np.asarray([[i] for i in range(nb_indices)])}
        max_reservoir_samples = 5
        max_reservoir_replacement_size = 5
        multiple = 10
        max_jobs_at_once = max_reservoir_samples
        wait_time_reservoir = 1.0
        wait_time_map = 0.1
        nb_map_workers = 2
        max_queue_size_pin = 4
        numpy_sequence = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        sequence = trw.train.SequenceAsyncReservoir(
            numpy_sequence,
            function_to_run=functools.partial(function_to_run_multiple_wait, nb=multiple, wait_time=wait_time_reservoir, text='reservoir_loaded'),
            max_reservoir_samples=max_reservoir_samples,
            min_reservoir_samples=max_reservoir_samples,
            max_jobs_at_once=max_jobs_at_once,
            max_reservoir_replacement_size=max_reservoir_replacement_size,
            reservoir_sampler=trw.train.SamplerSequential()).collate()
        
        sequence = sequence.rebatch(batch_size=1).map(functools.partial(function_to_run_id_wait, wait_time=0), nb_workers=nb_map_workers, max_queue_size_pin=max_queue_size_pin)

        nb_samples = 0
        expected_time = max_reservoir_samples * wait_time_map * (nb_epochs - 1) * multiple #/ max(1, nb_map_workers)
        time_start = None
        for epoch in range(nb_epochs):
            if epoch == 1:
                # discard first epoch timing due to processes/threads creation time
                time_start = time.perf_counter()

            epoch_start = time.perf_counter()
            for _, batch in enumerate(sequence):
                # simulate a workload. Overhead of the map
                # should be hidden!
                time.sleep(wait_time_map)
                nb_samples += trw.utils.len_batch(batch)

            # check the expected prefetch size: how many batches
            # are already processed and queued? 
            average_prefetch = sequence.debug_metadata.pin_queue_size / sequence.debug_metadata.nb_batches
            assert abs(max_queue_size_pin - 1 - average_prefetch) < 0.5, f'expected={max_queue_size_pin}, got={average_prefetch}'
            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            print(f'epoch={epoch}, epoch_time={epoch_time}')
        time_end = time.perf_counter()
        time_taken = time_end - time_start
        print(f'DONE, time_taken={time_taken}, expected_time={expected_time}')
        assert abs(time_taken - expected_time) < 0.5

        expected_samples = max_reservoir_samples * nb_epochs * multiple
        assert expected_samples == nb_samples, f'nb_samples={nb_samples}, expected_samples={expected_samples}'