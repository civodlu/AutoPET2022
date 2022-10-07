import collections
from typing import Optional, List

import torchvision
import os
import numpy as np
import torch
from .utils import get_data_root
from ..train import SequenceArray, SamplerRandom
from ..basic_typing import Datasets
from ..transforms import Transform


def create_cifar10_dataset(
        batch_size: int = 300,
        root: Optional[str] = None,
        transform_train: Optional[List[Transform]] = None,
        transform_valid: Optional[List[Transform]] = None,
        nb_workers: int = 2,
        data_processing_batch_size: int = None,
        normalize_0_1: bool = True) -> Datasets:

    root = get_data_root(root)

    if data_processing_batch_size is None:
        if nb_workers > 0:
            data_processing_batch_size = batch_size // nb_workers
        else:
            data_processing_batch_size = batch_size

    cifar_path = os.path.join(root, 'cifar10')

    train_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=True, transform=None, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=cifar_path, train=False, transform=None, download=True)

    def convert_image(dataset):
        normalization_factor = 1.0
        if normalize_0_1:
            normalization_factor = 255.0

        d = np.transpose(dataset.data.astype(np.float32), (0, 3, 1, 2)) / normalization_factor
        return torch.from_numpy(d)

    ds = {'images': convert_image(train_dataset), 'targets': np.asarray(train_dataset.targets, dtype=np.int64).reshape((-1, 1))}
    
    def create_sequence(transforms, ds):
        if transforms is None:
            sequence = SequenceArray(ds, SamplerRandom(batch_size=batch_size))
        else:
            assert batch_size % data_processing_batch_size == 0
            nb_jobs_for_one_batch = batch_size // data_processing_batch_size
            sampler = SamplerRandom(batch_size=data_processing_batch_size)
            sequence = SequenceArray(ds, sampler=sampler).map(transforms, nb_workers=nb_workers, max_jobs_at_once=1 * nb_workers, max_queue_size_pin=nb_jobs_for_one_batch)
            sequence = sequence.batch(batch_size // data_processing_batch_size)
        return sequence

    splits = collections.OrderedDict()
    splits['train'] = create_sequence(transform_train, ds).collate()

    ds = {'images': convert_image(test_dataset), 'targets': np.asarray(test_dataset.targets, dtype=np.int64).reshape((-1, 1))}
    splits['test'] = create_sequence(transform_valid, ds).collate()
    return {
        'cifar10': splits
    }
