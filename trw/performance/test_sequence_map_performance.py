"""
Assess the performance of the data pipeline
"""

import numpy as np
import time
import trw.train
import torch
import functools
from unittest import TestCase


def load_fake_volumes_npy(item):
    v = np.ndarray([512, 512, 500], dtype=np.float32)
    v.fill(float(item['values']))

    r = {
        'volume': v
    }
    return r


def load_fake_volumes_torch(item):
    v = torch.zeros([512, 512, 500], dtype=torch.float32)
    v.fill_(float(item['values']))

    r = {
        'volume': v
    }
    return r


def create_numpy(shape, value):
    v = np.ndarray(shape, dtype=np.float32)
    v.fill(value)
    return v


def create_torch(shape, value):
    v = torch.zeros(shape, dtype=torch.float32)
    v.fill_(value)
    return v


def identity(batch):
    return batch


def wait(batch, time_to_sleep=0.2):
    time.sleep(time_to_sleep)
    return batch


sleeping_time = 1.0


class TestDatasetLoaderPerf(TestCase):
    @staticmethod
    def map_async_20_numpy():
        # large numpy arrays, this is problematic: the data will be copied to the worker (i.e., this can be very slow for
        # large datasets)
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_npy, nb_workers=2)
        time.sleep(sleeping_time)

        time_create_start = time.time()
        load_fake_volumes_torch({'values': 1.0})
        time_create_end = time.time()

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        print('test_map_async_1.TIME=', time_end - time_start, 'time_single_load=', time_create_end - time_create_start, 'bo_async_time=', (time_create_end - time_create_start) * len(split_np['values']))

        assert len(vs) == 20
        return time_end - time_start

    @staticmethod
    def map_async_20_pytorch():
        # large pytorch arrays, this should be much faster to share compared to numpy arrays as
        # only descriptors are effectively sent through the queue
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_torch, nb_workers=2)
        time.sleep(sleeping_time)

        time_create_start = time.time()
        load_fake_volumes_torch({'values': 1.0})
        time_create_end = time.time()

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        print('test_map_async_1.TIME=', time_end - time_start, 'time_single_load=', time_create_end - time_create_start, 'bo_async_time=', (time_create_end - time_create_start) * len(split_np['values']))

        assert len(vs) == 20
        return time_end - time_start

    def test_large_pytorch_faster_numpy_transfers(self):
        # test that pytorch object are shared between processes and not copied as for
        # numpy arrays
        time_numpy = TestDatasetLoaderPerf.map_async_20_numpy()
        time_pytorch = TestDatasetLoaderPerf.map_async_20_pytorch()

        print('numpy=', time_numpy, 'pytorch=', time_pytorch)
        self.assertTrue(time_pytorch * 2 < time_numpy)

    @staticmethod
    def run_pipeline(augmentation_fn, batch, nb_workers=2, batch_size=1000):
        sampler = trw.train.sampler.SamplerRandom(batch_size=batch_size)
        split = trw.train.SequenceArray(batch, sampler=sampler).map(augmentation_fn, nb_workers=nb_workers)
        time.sleep(sleeping_time)

        print('STARTED')
        time_start = time.time()
        bs = []
        for b in split:
            bs.append(str(b))
        time_end = time.time()

        assert len(bs) == trw.train.len_batch(batch) / batch_size
        return time_end - time_start


    def test_pipeline_numpy(self):
        shape_small = [60000, 1]
        shape_large = [60000, 10000]
        nb_workers = 4

        data = {
            'images': create_numpy(shape_small, 42.0)
        }

        time_numpy_small = self.run_pipeline(identity, data, nb_workers=nb_workers)

        data = {
            'images': create_numpy(shape_large, 42.0)
        }

        time_numpy_large = self.run_pipeline(identity, data, nb_workers=nb_workers)
        print('TIME', time_numpy_small)


        data = {
            'images': create_torch(shape_small, 42.0)
        }

        time_torch_small = self.run_pipeline(identity, data, nb_workers=nb_workers)

        data = {
            'images': create_torch(shape_large, 42.0)
        }
        time_torch_large = self.run_pipeline(identity, data, nb_workers=nb_workers)
        print('TIME', time_numpy_small, time_numpy_large, time_torch_small, time_torch_large)

        # we expect to have almost the same timing for small/larget pytorch tensors
        self.assertTrue(time_torch_large < 1.2 * time_torch_small)

        # we expect to have large differences for the timing of small/larget numpy arrays
        self.assertTrue(time_numpy_large > 1.8 * time_numpy_small)


    def test_job_and_worker_processing_overlap(self):
        # The idea of the data pipeline is that we have data to process using the GPU (or `job`)
        # we also typically require data augmentation (or `worker` processes) that can be done
        # at the same time as the GPU calculations. This test measures what this overlap for
        # a dataset mimicking MNIST
        shape = [60000, 784]
        sub_batch_size = 1000
        full_batch_ratio = 5
        full_batch_size = full_batch_ratio * sub_batch_size
        time_to_sleep = 0.35
        full_batch_time_to_sleep = 0.5
        nb_batches = shape[0] / sub_batch_size
        nb_full_batch = shape[0] / full_batch_size
        nb_workers = 4

        data = {
            'images': create_torch(shape, 42.0)
        }

        sampler = trw.train.sampler.SamplerRandom(batch_size=sub_batch_size)
        split = trw.train.SequenceArray(data, sampler=sampler).map(functools.partial(wait, time_to_sleep=time_to_sleep), nb_workers=nb_workers, queue_timeout=0.001, max_jobs_at_once=10 * nb_workers).batch(full_batch_ratio)
        time.sleep(sleeping_time * 4)

        print('Started!!!')  # here we should have all processes started, else it will affect the timing

        time_start = time.time()
        bs = []
        for b in split:
            bs.append(b)
            time.sleep(full_batch_time_to_sleep)
        time_end = time.time()
        assert len(bs) == nb_full_batch, 'there is some error in calculation of `nb_full_batch`!'

        process_time = time_end - time_start  # this is what we want to optimize!
        expected_job_time = nb_full_batch * full_batch_time_to_sleep
        expected_preprocessing_time = nb_batches * time_to_sleep / nb_workers

        print('expected_preprocessing_time=', expected_preprocessing_time)
        print('expected_job_time=', expected_job_time)
        print('process_time', process_time)
        print('nb_full_batch', len(bs))
        print('overhead ratio=', (process_time - expected_job_time) / expected_job_time)
