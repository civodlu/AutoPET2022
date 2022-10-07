import time

import trw
from unittest import TestCase


keep_ids = set()


def fn_to_run_identity(data):
    return data


def fn_to_run_add_msg(data):
    data['message'] = 'done'
    return data


def post_process_results_fun(data, channel_main_to_worker, channel_worker_to_main):
    keep_ids.add(data)

    while channel_main_to_worker.poll():
        message = channel_main_to_worker.recv()
        assert message == 'test'
    assert len(keep_ids) == 1

    channel_worker_to_main.send('added test')


class TestJobExecutor(TestCase):
    def test_job_executor_simple_processing(self):
        """
        Process a series of jobs. Make sure all jobs are processed exactly once
        """
        executor = trw.train.job_executor2.JobExecutor2(nb_workers=2, function_to_run=fn_to_run_add_msg)

        # post jobs
        for i in range(10):
            success = False
            while not success:
                success = executor.put({'job': i})
                time.sleep(0.05)

        jobs_processes = set()
        nb_samples = 0
        while nb_samples < 10:
            metadata, r = executor.pin_memory_queue.get()
            jobs_processes.add(r['job'])
            print('RESULT=', r)
            assert r['message'] == 'done'
            nb_samples += 1

        executor.close()
        assert len(jobs_processes) == 10
