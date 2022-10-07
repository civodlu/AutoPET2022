import time
import collections
import os
from typing import Optional, List, Tuple

import torchvision
import numpy as np
import torch
from trw.train import SequenceArray, SamplerRandom

from ..basic_typing import ShapeX, Datasets, DatasetsInfo
from ..transforms import Transform
from .utils import get_data_root


def _clutter(images, cluttered_size, clutter_window, nb_clutter_windows, normalization_factor):
    images = images.float().numpy() / normalization_factor
    image_size = images.shape[1:]
    cluttered_images = np.zeros([len(images), cluttered_size[0], cluttered_size[1]])

    start = time.perf_counter()
    for n in range(len(images)):
        for nn in range(nb_clutter_windows):
            i = np.random.randint(low=0, high=len(images) - 1)
            sy = np.random.randint(low=0, high=image_size[0] - clutter_window[0] - 1)
            sx = np.random.randint(low=0, high=image_size[1] - clutter_window[1] - 1)

            syc = np.random.randint(low=0, high=cluttered_size[0] - clutter_window[0] - 1)
            sxc = np.random.randint(low=0, high=cluttered_size[1] - clutter_window[1] - 1)
            cluttered_images[n, syc:syc+clutter_window[0], sxc:sxc+clutter_window[1]] = images[i, sy:sy+clutter_window[0], sx:sx+clutter_window[1]]

        y = np.random.randint(low=0, high=cluttered_size[0] - image_size[0] - 1)
        x = np.random.randint(low=0, high=cluttered_size[1] - image_size[1] - 1)
        cluttered_images[n, y:y + image_size[0], x:x + image_size[1]] = images[n, :, :]

    end = time.perf_counter()
    print('preprocessing time=', end - start)

    return torch.from_numpy(np.float32(cluttered_images)).view((len(images), 1, cluttered_size[0], cluttered_size[1]))


def create_mnist_cluttered_datasset(
        batch_size: int = 1000,
        cluttered_size: ShapeX = (64, 64),
        clutter_window: ShapeX = (6, 6),
        nb_clutter_windows: int = 16,
        root: Optional[str] = None,
        train_transforms: List[Transform] = None,
        test_transforms: List[Transform] = None,
        nb_workers: int = 5,
        data_processing_batch_size: int = 200,
        normalize_0_1: bool = False) -> Tuple[Datasets, DatasetsInfo]:
    """

    Args:
        batch_size:
        cluttered_size: the size of the final image
        root:
        clutter_window: the size of the random windows to create the clutter
        nb_clutter_windows: the number of clutter windows added to the image
        train_transforms: the transform function applied on the training batches
        test_transforms: the transform function applied on the test batches
        nb_workers: the number of workers to preprocess the dataset
        data_processing_batch_size: the number of samples each worker process at once
        normalize_0_1: if `True`, the pixels will be in range [0..1]

    Returns:
        datasets
    """
    root = get_data_root(root)

    train_dataset = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=True)

    test_dataset = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=True)

    normalization_factor = 1.0
    if normalize_0_1:
        normalization_factor = 255.0

    train_images = _clutter(
        train_dataset.data,
        cluttered_size,
        clutter_window,
        nb_clutter_windows,
        normalization_factor)

    train_split = {
        'images': train_images,
        'targets': train_dataset.targets
    }

    if train_transforms is None:
        train_sequence = SequenceArray(train_split, SamplerRandom(batch_size=batch_size))
    else:
        assert batch_size % data_processing_batch_size == 0
        sampler = SamplerRandom(batch_size=data_processing_batch_size)
        train_sequence = SequenceArray(train_split, sampler=sampler).map(
            train_transforms,
            nb_workers=nb_workers,
            max_jobs_at_once=nb_workers * 2)
        train_sequence = train_sequence.batch(batch_size // data_processing_batch_size)

    test_images = _clutter(
        test_dataset.data,
        cluttered_size,
        clutter_window,
        nb_clutter_windows,
        normalization_factor)

    test_split = {
        'images': test_images,
        'targets': test_dataset.targets
    }

    if test_transforms is None:
        test_sequence = SequenceArray(test_split, SamplerRandom(batch_size=batch_size))
    else:
        assert batch_size % data_processing_batch_size == 0
        sampler = SamplerRandom(batch_size=data_processing_batch_size)
        test_sequence = SequenceArray(test_split, sampler=sampler).map(
            train_transforms,
            nb_workers=nb_workers,
            max_jobs_at_once=nb_workers * 2)
        test_sequence = test_sequence.batch(batch_size // data_processing_batch_size)

    # generate the class mapping
    mapping = dict()
    mappinginv = dict()
    for id, name in enumerate(torchvision.datasets.MNIST.classes):
        mapping[name] = id
        mappinginv[id] = name
    output_mappings = {'targets': {'mapping': mapping, 'mappinginv': mappinginv}}

    datasets_info = {
        'mnist': {
            'train': {'output_mappings': output_mappings},
            'test': {'output_mappings': output_mappings},
        }
    }

    splits = dict()
    splits['train'] = train_sequence
    splits['test'] = test_sequence

    datasets = {
        'mnist': splits
    }

    return datasets, datasets_info
