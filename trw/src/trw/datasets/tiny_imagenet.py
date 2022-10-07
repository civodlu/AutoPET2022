import collections
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from ..train import SequenceArray, SamplerSequential, SamplerRandom
from ..basic_typing import Datasets, DatasetsInfo
from ..transforms import Transform, TransformCompose
from .utils import get_data_root


class TinyImageNet(Dataset):
    """
    Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Notes:
        The `test` valid is discarded since we do not have the test labels
    """
    def __init__(self, root, split='train', num_images_per_class=500):
        """

        Args:
            root:
            split: a split name. One of train, val
            num_images_per_class:
        """

        assert num_images_per_class <= 500
        self.root = os.path.expanduser(root)
        self.split = split
        self.split_dir = os.path.join(root, self.split)
        self.labels = []
        self.images = []

        # build class label - number mapping
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(num_images_per_class):
                    file = f'{label_text}_{cnt}.JPEG'
                    path = os.path.join(root, 'train', label_text, 'images', file)

                    self.images.append(TinyImageNet.read_image(path))
                    self.labels.append(i)

        elif self.split == 'val':
            num_classes = len(self.label_texts)
            with open(os.path.join(self.split_dir, 'val_annotations.txt'), 'r') as fp:
                for line_number, line in enumerate(fp.readlines()):
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]

                    path = os.path.join(root, 'val', 'images', file_name)
                    self.images.append(TinyImageNet.read_image(path))
                    self.labels.append(self.label_text_to_number[label_text])

                    if len(self.images) >= num_classes * num_images_per_class:
                        break

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    @staticmethod
    def read_image(path):
        # read an RGB image, transpose the color channel so that we have format CHW
        img = np.array(Image.open(path))
        if len(img.shape) == 2:
            return np.stack([img, img, img])
        else:
            return img.transpose((2, 0, 1))


def create_tiny_imagenet_dataset(
        batch_size: int,
        num_images_per_class: int = 500,
        transforms_train: List[Transform] = None,
        transforms_valid: List[Transform] = None,
        nb_workers: int = 4,
        root: Optional[str] = None) -> Tuple[Datasets, DatasetsInfo]:

    root = get_data_root(root)
    root_imagenet = os.path.join(root, 'tiny_imagenet/tiny-imagenet-200')
    dataset_train = TinyImageNet(root_imagenet, split='train', num_images_per_class=num_images_per_class)
    dataset_valid = TinyImageNet(root_imagenet, split='val', num_images_per_class=num_images_per_class)

    def create_split(dataset, transform_fn, is_train):
        images = torch.from_numpy(np.asarray(dataset.images))
        targets = torch.from_numpy(np.asarray(dataset.labels)).long()
        split = collections.OrderedDict([
            ('images', images),
            ('targets', targets),
        ])

        if is_train:
            sampler = SamplerRandom(batch_size=batch_size)
        else:
            sampler = SamplerSequential(batch_size=batch_size)
        split = SequenceArray(split, sampler=sampler)

        if transform_fn is not None:
            transform_fn = TransformCompose(transforms=transform_fn)
            if is_train:
                split = split.async_reservoir(
                    max_reservoir_samples=300,
                    function_to_run=transform_fn,
                    min_reservoir_samples=100,
                    nb_workers=nb_workers,
                    max_jobs_at_once=100,
                    max_reservoir_replacement_size=50).collate()
            else:
                split = split.map(transform_fn, nb_workers=0)
        return split

    splits = dict([
        ('train', create_split(dataset_train, transforms_train, is_train=True)),
        ('valid', create_split(dataset_valid, transforms_valid, is_train=False))
    ])

    mapping = dataset_train.label_text_to_number
    mappinginv = {n: name for name, n in mapping.items()}
    output_mappings = {'targets': {'mapping': mapping, 'mappinginv': mappinginv}}

    datasets = {'tiny_imagenet_200': splits}
    datasets_info = {
        'tiny_imagenet_200': {
            'train': {'output_mappings': output_mappings},
            'valid': {'output_mappings': output_mappings},
        }
    }

    return datasets, datasets_info

