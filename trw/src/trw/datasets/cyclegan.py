import os
from functools import partial
from typing import Optional, List

import torch
import numpy as np
from typing_extensions import Literal
from ..basic_typing import Datasets
from ..train import SamplerRandom, SequenceArray, SamplerSequential
from ..datasets.utils import download_and_extract_archive
from ..transforms import Transform, TransformCompose, TransformResize, TransformRandomFlip, TransformRandomCropPad
from .utils import get_data_root
import glob
from PIL import Image


cycle_gan_dataset = Literal[
    'ae_photos',
    'apple2orange',
    'cezanne2photo',
    'cityscapes',
    'facades',
    'grumpifycat',
    'horse2zebra',
    'iphone2dslr_flower',
    'maps',
    'mini',
    'mini_colorization',
    'mini_pix2pix',
    'monet2photo',
    'summer2winter_yosemite',
    'ukiyoe2photo',
    'vangogh2photo'
]


def image_to_torch(i):
    return torch.from_numpy(np.array(i).transpose((2, 0, 1))).unsqueeze(0)


def load_case(batch, dataset_a, dataset_b, transform, aligned):
    case_ids = batch['case_id']

    batch_a = dataset_a[case_ids]
    if aligned:
        batch_b = dataset_b[case_ids]
    else:
        batch_b = np.random.choice(dataset_b, len(batch_a), replace=True)

    images_a = []
    images_b = []
    for n in range(len(batch_a)):
        image_a = np.asarray(Image.open(batch_a[n]).convert('RGB'))
        image_b = np.asarray(Image.open(batch_b[n]).convert('RGB'))
        images_a.append(image_to_torch(image_a))
        images_b.append(image_to_torch(image_b))

    data_batch = {
        'case_id': case_ids,
        'image_a': torch.cat(images_a),
        'image_b': torch.cat(images_b),
    }

    if transform is not None:
        data_batch = transform(data_batch)

    return data_batch


def train_transform():
    return [
        TransformResize(size=[286, 286]),
        TransformRandomFlip(axis=3),
        TransformRandomCropPad(padding=None, shape=[3, 256, 256]),
    ]


def create_cycle_gan_dataset(
        dataset_name: cycle_gan_dataset,
        batch_size: int = 32,
        root: Optional[str] = None,
        url: str = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/',
        transform_train: Optional[List[Transform]] = None,
        transform_valid: Optional[List[Transform]] = None,
        aligned: bool = False,
        loading_batch_size: int = 4,
        nb_workers: int = 4) -> Datasets:
    """
    Datasets used for image to image translation (domain `A` to domain `B`).

    Args:
        dataset_name: the name of the dataset
        batch_size: the size of each batch
        root: the root path where to store the dataset
        url: specify the URL from which the dataset is downloaded
        transform_train: transform applied to train dataset
        transform_valid: transform applied to valid dataset
        aligned: if True, the images `A` and `B` will be considered aligned. If False,
            `B` will be randomly sampled from the list of available images in `B`
        nb_workers: the number of workers to process the images
        loading_batch_size: the number of images loaded by a worker

    Returns:
        a dataset
    """
    root = get_data_root(root)

    if transform_train is not None:
        assert isinstance(transform_train, list)
        transform_train = TransformCompose(transform_train)  # type: ignore

    if transform_valid is not None:
        assert isinstance(transform_valid, list)
        transform_valid = TransformCompose(transform_valid)  # type: ignore

    url_dataset = os.path.join(url, dataset_name + '.zip')
    dataset_path = os.path.join(root, dataset_name + '_cycle')
    download_and_extract_archive(url_dataset, dataset_path)

    def create_split(is_training, path, transform):
        if is_training:
            sampler = SamplerRandom(batch_size=loading_batch_size)
        else:
            sampler = SamplerSequential(batch_size=loading_batch_size)

        split = 'train' if is_training else 'test'
        images_a = os.path.join(path, dataset_name, split + 'A')
        images_a = glob.glob(os.path.join(images_a, '*.jpg'))
        images_a = np.asarray(sorted(images_a))

        images_b = os.path.join(path, dataset_name, split + 'B')
        images_b = glob.glob(os.path.join(images_b, '*.jpg'))
        images_b = np.asarray(sorted(images_b))

        if aligned:
            assert len(images_a) == len(images_b)
            for n in range(len(images_a)):
                name_a = os.path.basename(images_a[n]).replace('_A', '').replace('_B', '')
                name_b = os.path.basename(images_b[n]).replace('_A', '').replace('_B', '')
                assert name_a == name_b

        sequence = SequenceArray({'case_id': np.arange(len(images_a))}, sampler=sampler)
        sequence = sequence.map(partial(load_case,
                                        dataset_a=images_a,
                                        dataset_b=images_b,
                                        transform=transform,
                                        aligned=aligned),
                                nb_workers=nb_workers
                                )
        sequence = sequence.rebatch(batch_size)
        return sequence

    dataset = {
        'train': create_split(is_training=True, path=dataset_path, transform=transform_train),
        'valid': create_split(is_training=False, path=dataset_path, transform=transform_valid),
    }

    return {
        dataset_name: dataset
    }
