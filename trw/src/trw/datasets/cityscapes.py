from typing import Optional, List

import torch
import torchvision
import numpy as np
from ..basic_typing import Datasets
from ..train import SequenceArray
from ..train import SamplerRandom, SamplerSequential
from .utils import get_data_root
import functools
import collections
import os

from ..transforms import Transform
from typing_extensions import Literal


def image_to_torch(i):
    return torch.from_numpy(np.array(i).transpose((2, 0, 1))).unsqueeze(0)


def segmentation_to_torch(i):
    return torch.from_numpy(np.array(i)).type(torch.int64).unsqueeze(0).unsqueeze(0)


def load_case(batch, dataset, transform):
    case_ids = batch['case_id']

    images = []
    segmentations = []

    for case_id in case_ids:
        image, segmentation = dataset[case_id]
        images.append(image_to_torch(image))
        segmentations.append(segmentation_to_torch(segmentation))

    data_batch = {
        'case_id': case_ids,
        'image': torch.cat(images),
        'segmentation': torch.cat(segmentations)
    }

    if transform is not None:
        data_batch = transform(data_batch)

    return data_batch


def create_cityscapes_dataset(
        batch_size: int = 32,
        root: Optional[str] = None,
        transform_train: Optional[List[Transform]] = None,
        transform_valid: Optional[List[Transform]] = None,
        nb_workers: int = 4,
        target_type: Literal['semantic'] = 'semantic') -> Datasets:
    """
    Load the cityscapes dataset. This requires to register on their website https://www.cityscapes-dataset.com/
    and manually download the dataset.


    The dataset is composed of 3 parts: gtCoarse, gtFine, leftImg8bit. Download each package and unzip in a
    folder (e.g., `cityscapes`)

    Args:
        batch_size:
        root: the folder containing the 3 unzipped cityscapes data `gtCoarse`, `gtFine`, `leftImg8bit`
        transform_train: the transform to apply on the training batches
        transform_valid: the transform to apply on the validation batches
        nb_workers: the number of workers for each split allocated to the data loading and processing
        target_type: the segmentation task

    Returns:
        a dict of splits. Each split is a :class:`trw.train.Sequence`
    """
    root = get_data_root(root)

    cityscapes_path = os.path.join(root, 'cityscapes')
    train_dataset = torchvision.datasets.cityscapes.Cityscapes(cityscapes_path, mode='fine', split='train', target_type=target_type)
    valid_dataset = torchvision.datasets.cityscapes.Cityscapes(cityscapes_path, mode='fine', split='val', target_type=target_type)

    train_sampler = SamplerRandom(batch_size=batch_size)
    train_sequence = SequenceArray({'case_id': np.arange(len(train_dataset))}, sampler=train_sampler)
    train_sequence = train_sequence.map(
        functools.partial(load_case, dataset=train_dataset, transform=transform_train), nb_workers=nb_workers)

    valid_sampler = SamplerSequential(batch_size=batch_size)
    valid_sequence = SequenceArray({'case_id': np.arange(len(valid_dataset))}, sampler=valid_sampler)
    valid_sequence = valid_sequence.map(
        functools.partial(load_case, dataset=valid_dataset, transform=transform_valid), nb_workers=nb_workers)

    dataset = collections.OrderedDict([
        ('train', train_sequence),
        ('valid', valid_sequence)
    ])

    return collections.OrderedDict([
        ('cityscapes', dataset)
    ])
