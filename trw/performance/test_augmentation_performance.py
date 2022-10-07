from unittest import TestCase
import trw
import time
import numpy as np


class TestTransform(TestCase):
    def test_augmented_split_performance(self):
        # mimic MNSIT shapes
        N = 60000
        data = np.random.rand(N, 1, 28, 28).astype(np.float)
        c = np.zeros([N], dtype=int)
        split_np = {'images': data, 'classes' :c}

        split_no_transform = trw.train.SequenceArray(split_np)
        transform = trw.transforms.TransformRandomCrop(padding=[0, 8, 8])
        split_with_transform = trw.train.SequenceArray(split_np, transforms=transform, sampler=trw.train.SamplerRandom(batch_size=500))

        time_raw_augmentation_start = time.time()
        _ = transform(split_np)
        time_raw_augmentation_end = time.time()
        time_raw_augmentation = time_raw_augmentation_end - time_raw_augmentation_start
        print('TIME transform alone=', time_raw_augmentation)

        time_no_augmentation_start = time.time()
        for _ in split_no_transform:
            pass
        time_no_augmentation_end = time.time()
        time_no_augmentation = time_no_augmentation_end - time_no_augmentation_start
        print('TIME no augmentation=', time_no_augmentation)

        time_with_augmentation_start = time.time()
        for _ in split_with_transform:
            pass
        time_with_augmentation_end = time.time()
        time_with_augmentation = time_with_augmentation_end - time_with_augmentation_start
        print('TIME with augmentation=', time_with_augmentation)

        split_with_transform_last_main_thread = trw.train.SequenceArray(split_np, sampler=trw.train.SamplerRandom(batch_size=500)).map(transform)
        time_split_with_transform_last_main_start = time.time()
        for _ in split_with_transform_last_main_thread:
            pass
        time_split_with_transform_last_main_end = time.time()
        time_split_with_transform_last_main = time_split_with_transform_last_main_end - time_split_with_transform_last_main_start
        print('TIME with augmentation last main thread=', time_split_with_transform_last_main)

        split_with_transform_last_workers_thread = trw.train.SequenceArray(split_np, sampler=trw.train.SamplerRandom(batch_size=25)).map(transform, nb_workers=2, max_jobs_at_once=5).batch(20)
        for _ in split_with_transform_last_workers_thread:
            break  # we MUST exclude the process creation time

        time_split_with_transform_last_workers_start = time.time()
        for _ in split_with_transform_last_workers_thread:
            pass
        time_split_with_transform_last_workers_end = time.time()
        time_split_with_transform_last_workers = time_split_with_transform_last_workers_end - time_split_with_transform_last_workers_start
        print('TIME with augmentation last main thread=', time_split_with_transform_last_workers)
