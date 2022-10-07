from functools import partial
from pprint import PrettyPrinter
import numpy as np
import time
import datetime
import trw.train
import torch
from unittest import TestCase
import os
import copy

from trw.callbacks.callback_debug_processes import log_all_tree


START_TIME = time.time()


import matplotlib
matplotlib.use('Agg')  # non interactive backend


def load_fake_volumes_npy(item):
    v = np.ndarray([32, 32, 64], dtype=np.float32)
    v.fill(float(item['values']))

    r = {
        'volume': v
    }
    return r


def load_fake_volumes_small_npy(item):
    v = np.ndarray([1, 1, 2], dtype=np.float32)
    v.fill(float(item['values']))

    time.sleep(0.5)
    r = {
        'volume': v
    }
    return r


def load_fake_volumes_torch(item):
    time_start = time.time()
    print('torch_fake_before')

    v = torch.ones([32, 32, 64], dtype=torch.float32)
    print('torch_fake_created')
    v.fill_(float(item['values']))
    print('torch_fake_filled')
    time_end = time.time()
    print('torch_construction_time=', time_end - time_start)
    r = {
        'volume': v
    }
    print('torch_fake_done!')
    return r


def load_data(item, time_sleep=0.2):
    item['time_created'] = time.time()
    time.sleep(time_sleep)
    item['time_loaded'] = time.time()
    #print('loading data | ', os.getpid(), item['indices'], time.time() - START_TIME)
    return item


def create_value(item, time_sleep=0.1):
    assert len(item) == 1
    time.sleep(time_sleep)
    item[0]['value'] = item[0]['indices']
    #print('create_value | ', os.getpid(), item[0]['indices'], time.time() - item[0]['time_created'])
    return item


def load_data_or_generate_error(item, error_index=10):
    print('job | ', os.getpid(), ' | loading data |', item['indices'], datetime.datetime.now().time())
    item['time_created'] = time.time()
    time.sleep(0.01)

    if item['indices'] == error_index:
        raise IndexError('This is an expected exception to test worker recovery from failure!')

    item['time_loaded'] = time.time()
    return item


def load_data_or_return_none(item, error_index=10):
    print('job | ', os.getpid(), ' | loading data |', item['indices'], datetime.datetime.now().time())
    item['time_created'] = time.time()
    time.sleep(0.01)

    if item['indices'] == error_index:
        return None

    item['time_loaded'] = time.time()
    return item


def run_augmentations(item):
    print('job | ', os.getpid(), ' | augmentation data', item['indices'], datetime.datetime.now().time())
    time.sleep(0.1)
    item['time_augmented'] = time.time()
    items = []
    for i in range(10):
        item_copy = copy.deepcopy(item)
        item_copy['augmentation'] = i
        items.append(item_copy)
    return items


def load_5_items(item):
    items = []
    for i in range(5):
        v = torch.ones([512, 512, 1], dtype=torch.float32)
        v.fill_(float(item['values']))
        items.append({'volume': v})
    return items


def double_values(item):
    item['values'] = item['values'] * 2.0
    return item


class TestSequenceMap(TestCase):
    def test_a_map_async_20_pytorch(self):
        # TODO in some instances, depending on the test position (e.g., at the end rather than
        #   at the beginning of the tests), pytorch can deadlock between
        #   `torch_fake_before` and `torch_fake_created` statements. This seems to be related
        #   to the openmp library used
        #   see: https://github.com/pytorch/pytorch/issues/17199
        #        https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58378
        torch.set_num_threads(1)

        # large pytorch arrays, this should be much faster to share compare to numpy arrays as
        # only descriptors are effectively sent through the queue
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_torch, nb_workers=1, max_jobs_at_once=1)

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        print('test_map_async_1.TIME=', time_end - time_start)

        assert(len(vs) == 20)
        split.close()
        print('DONE')

    def test_map_async_10(self):
        # create very large numpy arrays and send it through multiprocessing.Queue: this is expected to be slow!
        split_np = {'values': np.arange(0, 10)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_small_npy, nb_workers=1)

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print('SEQUENCE Item=', v['volume'].shape)
        time_end = time.time()
        print('test_map_async_1.TIME=', time_end - time_start)
        split.job_executor.job_report()
        split.close()
        assert len(vs) == 10, f'got={len(vs)}'

    def test_map_async_20(self):
        # create very large numpy arrays and send it through multiprocessing.Queue: this is expected to be slow!
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_npy, nb_workers=2)

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        split.close()
        print('test_map_async_1.TIME=', time_end - time_start)
        assert(len(vs) == 20)

    def test_map_async_sequence_interrupted(self):
        # create very large numpy arrays and send it through multiprocessing.Queue: this is expected to be slow!
        split_np = {'values': np.arange(100, 105)}
        split = trw.train.SequenceArray(split_np).map(load_fake_volumes_npy, nb_workers=2)

        for _ in split:
            # interrupt the sequence: here we have jobs populated that SHOULD not be
            # used in the next sequence iteration
            time.sleep(0.2)
            print('----------------BREAK----------')
            break

        time_start = time.time()
        vs = []
        for v in split:
            vs.append(v['volume'].shape)
            print(v['volume'].shape)
        time_end = time.time()
        split.close()
        print('test_map_async_1.TIME=', time_end - time_start, 'nb=', len(vs))
        assert(len(vs) == 5)

    def test_map_sync_multiple_items(self):
        # make sure we have iterate through each item of the returned items
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_5_items, nb_workers=0)

        vs = []
        for v in split:
            vs.append(v['volume'].shape)

        assert(len(vs) == 20 * 5)
        split.close()
        print('DONE')

    def test_map_async_multiple_items(self):
        # make sure we have iterate through each item of the returned items
        split_np = {'values': np.arange(100, 120)}
        split = trw.train.SequenceArray(split_np).map(load_5_items, nb_workers=3)

        vs = []
        for v in split:
            vs.append(v['volume'].shape)

        assert(len(vs) == 20 * 5)
        split.close()
        print('DONE')

    @staticmethod
    def run_complex_2_map(nb_workers, nb_indices, with_interruption):
        # the purpose of this test is to combine 2 maps: one executing slow calls
        # (e.g., IO limited) with another one to do augmentation (e.g., CPU limited)

        print('run_complex_2_map START', datetime.datetime.now().time())

        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).\
            map(load_data, nb_workers=nb_workers).\
            map(run_augmentations, nb_workers=1, max_jobs_at_once=None)

        # process creation is quite slow on windows (>0.7s), so create the processes first
        # so that creation time is not included in processing time
        if with_interruption:
            for batch in split:
                break

            print('----------ABORTED sequence')
            time.sleep(3)

        print('STARTED', datetime.datetime.now().time())
        time_start = time.time()
        batches = []
        ids = set()
        for batch in split:
            # time_processed = time.time()
            # processed_time = time_processed - batch['time_augmented']
            # loaded_time = time_processed - batch['time_loaded']
            # created_time = time_processed - batch['time_created']
            # batch['time_processed'] = time_processed
            # print(batch)
            # print('TIME----', processed_time, loaded_time, created_time, 'NOW=', datetime.datetime.now().time())
            batches.append(batch)
            ids.add(str(batch['indices'][0]) + '_' + str(batch['augmentation']))

        print('ENDED', datetime.datetime.now().time())

        expected_time = nb_indices * 2.0 / nb_workers + nb_indices * 0.1
        time_end = time.time()
        total_time = time_end - time_start
        print('total_time', total_time, 'Target time=', expected_time, 'nb_jobs=', len(ids), 'nb_jobs_expected=', len(indices) * 10)

        print(len(ids), len(indices) * 10)
        print(split.jobs_processed)
        print(split.jobs_queued)
        print(split.job_executor.jobs_queued)
        print(split.job_executor.jobs_processed.value)
        assert len(batches) == len(indices) * 10, f'nb_batches={len(batches)}, nb_indices={len(indices) * 10}'
        assert len(ids) == len(indices) * 10

        split.close()

        print('run_complex_2_map END', datetime.datetime.now().time())


    def test_complex_2_map__single_worker(self):
        TestSequenceMap.run_complex_2_map(1, nb_indices=10, with_interruption=False)

    def test_complex_2_map__single_worker_with_interruption(self):
        TestSequenceMap.run_complex_2_map(1, nb_indices=10, with_interruption=True)

    def test_complex_2_map__5_worker(self):
        TestSequenceMap.run_complex_2_map(5, nb_indices=10, with_interruption=False)

    def test_complex_2_map__5_worker_40(self):
        TestSequenceMap.run_complex_2_map(5, nb_indices=21, with_interruption=False)

    def test_split_closing(self):
        # make sure we can close the processes gracefully
        nb_indices = 40
        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).\
            map(load_data, nb_workers=4).\
            map(run_augmentations, nb_workers=1, max_jobs_at_once=None)
        for batch in split:
            break
        split.job_executor.close()

    def test_job_error_0_worker(self):
        nb_indices = 40
        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        indices = []
        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(load_data_or_generate_error, nb_workers=0) #.map(run_augmentations, nb_workers=1, max_jobs_at_once=None)
        for batch in split:
            index = batch['indices'][0]
            indices.append(index)

        assert 10 not in indices, 'index 10 should have failed!'
        assert len(indices) == nb_indices - 1

    def test_job_error_1_worker(self):
        nb_indices = 15
        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        indices = []
        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(
            load_data_or_generate_error,
            nb_workers=1)  # .map(run_augmentations, nb_workers=1, max_jobs_at_once=None)
        for batch_id, batch in enumerate(split):
            index = batch['indices'][0]
            indices.append(index)
            with split.job_executor.jobs_processed.get_lock():
                print('nb_queued=', split.job_executor.jobs_queued, 'nb_processed=', split.job_executor.jobs_processed.value)
                print('(SEQUENCE))=', split.jobs_processed, 'Batchid=', batch_id)
                print(index)

        print('DONE')
        del split

        print(indices)
        assert 10 not in indices, 'index 10 should have failed!'
        assert len(indices) == nb_indices - 1

    def test_single_job_exception(self):
        nb_indices = 40
        split = {
            'indices':  np.asarray(list(range(nb_indices))),
        }
        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(
            load_data_or_generate_error,
            nb_workers=0)

        indices = []
        for batch in split:
            index = batch['indices'][0]
            indices.append(index)
        assert 10 not in indices, 'index 10 should have failed!'
        assert len(indices) == nb_indices - 1

    def test_single_job_result_none(self):
        nb_indices = 40
        split = {
            'indices':  np.asarray(list(range(nb_indices))),
        }
        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(
            load_data_or_return_none,
            nb_workers=0)

        indices = []
        for batch in split:
            index = batch['indices'][0]
            indices.append(index)
        assert 10 not in indices, 'index 10 should have failed!'
        assert len(indices) == nb_indices - 1

    def test_one_worker_job_result_none(self):
        nb_indices = 40
        split = {
            'indices':  np.asarray(list(range(nb_indices))),
        }
        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential()).map(
            load_data_or_return_none,
            nb_workers=1)

        indices = []
        for batch in split:
            index = batch['indices'][0]
            indices.append(index)
        assert 10 not in indices, 'index 10 should have failed!'
        print(indices)
        assert len(indices) == nb_indices - 1, f'expected={nb_indices - 1}, got={len(indices)}'
        print('Done')

    def test_reservoir_map(self):
        # test various "workloads". Make sure pipeline doesn't deadlock
        np.random.seed(0)
        for i in range(10):
            print(f'---------- {i} -------------')
            time_sleep_1 = np.random.uniform(0.001, 0.5)
            time_sleep_2 = np.random.uniform(0.001, 0.5)
            nb_jobs_at_once = int(np.random.uniform(1, 5))
            nb_indices = int(np.random.uniform(10, 40))
            nb_workers = int(np.random.uniform(1, 4))
            nb_epochs = int(np.random.uniform(1, 7))

            split = {
                'indices': np.asarray(list(range(nb_indices))),
            }
            split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
            split = split.async_reservoir(max_reservoir_samples=nb_indices, function_to_run=partial(load_data, time_sleep=time_sleep_1), max_jobs_at_once=nb_jobs_at_once)
            split = split.map(partial(create_value, time_sleep=time_sleep_2), nb_workers=nb_workers)
            nb = 0
            for epoch in range(nb_epochs):
                print('Epoch=', epoch)
                for b in split:
                    if nb == 1:
                        logs = log_all_tree()
                        pp = PrettyPrinter(width=300)
                        pp.pprint(logs)

                    nb += 1

    def test_multiple_epochs_job_failures(self):
        nb_indices = 20
        nb_epochs = 50
        indices = np.asarray(list(range(nb_indices)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        split = split.map(load_data_or_generate_error, nb_workers=5, max_jobs_at_once=2)
        for e in range(nb_epochs):
            print(f'epoch={e}')
            indices = []
            for batch in split:
                index = batch['indices'][0]
                indices.append(index)
            assert 10 not in indices, 'index 10 should have failed!'
            assert len(indices) == nb_indices - 1

    def test_close(self):
        # make sure the .close() closes the job executor
        # with all it associated resource
        indices = np.asarray(list(range(20)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        split_map = split.map(partial(load_data, time_sleep=0), nb_workers=1)
        split = split_map.max_samples(1)
        split = split.rebatch(1)
        split = split.sub_batch(1)
        split = split.collate()

        all_items = []
        for n in range(len(indices)):
            for i in split:
                all_items.append(i['indices'][0])

        assert len(all_items) == len(indices)
        for i, item in enumerate(all_items):
            assert int(item) == i

        assert not split_map.job_executor.synchronized_stop.is_set()
        split.close()
        assert split_map.job_executor.synchronized_stop.is_set()

    def test_jobexecutor_process_killed(self):
        """
        Simulate a crash of a job executor process
        we expect the job executor to restart a new process
        and continue the processing, possibly losing one job result
        """
        indices = np.asarray(list(range(5)))
        split = {
            'indices': indices,
        }

        split = trw.train.SequenceArray(split, sampler=trw.train.SamplerSequential())
        split_map = split.map(partial(load_data, time_sleep=1), nb_workers=1)
        time.sleep(2.0)

        print('Starting sequence [process will be killed]')
        b_list = []
        for b_n, b in enumerate(split_map):
            if b_n == 1:
                print('killing process...')
                time.sleep(0.1)
                split_map.job_executor.processes[0].terminate()
            b_list.append(b)
            print(b)
        assert len(b_list) >= len(indices) - 2

        print('Starting sequence [no error]')
        b_list = []
        for b_n, b in enumerate(split_map):
            b_list.append(b)
            print(b)
        assert len(b_list) == len(indices)

        print('sequence done!')