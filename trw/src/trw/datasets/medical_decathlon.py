import collections
import functools
import os
from typing import Dict, Union, Callable, Optional, Mapping, MutableMapping

from ..basic_typing import Batch, Dataset, Datasets
from ..utils import optional_import
from .utils import download_and_extract_archive
import json
import numpy as np
import torch
from ..train import SequenceArray, SamplerSequential, SamplerRandom
from ..transforms import Transform
from .utils import get_data_root

nib = optional_import('nibabel')


def load_nifti(
        path: str,
        dtype,
        base_name: str,
        remove_patient_transform: bool = False) -> Dict[str, Union[str, torch.Tensor]]:
    """
    Load a nifti file and metadata.

    Args:
        path: the path to the nifti file
        base_name: the name of this data
        dtype: the type of the nifti image to be converted to
        remove_patient_transform: if ``True``, remove the affine transformation attached to the voxels

    Returns:
        a dict of attributes
    """
    data: Dict[str, Union[str, torch.Tensor]] = collections.OrderedDict()
    img = nib.load(path)
    if not remove_patient_transform:
        data[base_name + 'affine'] = torch.from_numpy(img.affine).unsqueeze(0)  # add the N component

    voxels = np.array(img.get_fdata(dtype=np.float32))
    if dtype != np.float32:
        voxels = voxels.astype(dtype)

    voxels = torch.from_numpy(voxels)
    voxels = voxels.unsqueeze(0).unsqueeze(0)
    data[base_name + 'voxels'] = voxels

    _, case_name = os.path.split(path)
    case_name = case_name.replace('.nii.gz', '')
    data['case_name'] = case_name

    return data


class MedicalDecathlonDataset:
    resource = {
        "Task01_BrainTumour": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
        "Task02_Heart": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar",
        "Task03_Liver": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
        "Task04_Hippocampus": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar",
        "Task05_Prostate": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar",
        "Task06_Lung": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task06_Lung.tar",
        "Task07_Pancreas": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar",
        "Task08_HepaticVessel": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task08_HepaticVessel.tar",
        "Task09_Spleen": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        "Task10_Colon": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task10_Colon.tar",
    }

    dataset_name = 'decathlon'

    def __init__(self, task_name: str, root: str, collection: str = 'training', remove_patient_transform: bool = False):
        task_resource = self.resource.get(task_name)
        if task_resource is None:
            raise RuntimeError(f'Task={task_name} does not exist!')
        root_data = os.path.join(root, self.dataset_name, task_name)
        download_and_extract_archive(task_resource, root_data)
        dataset_metadata_path = os.path.join(root_data, task_name, 'dataset.json')
        self.collection = collection

        with open(dataset_metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.task_root = os.path.join(root_data, task_name)
        self.remove_patient_transform = remove_patient_transform

    def __call__(self, id: int) -> MutableMapping[str, Union[str, torch.Tensor]]:
        data: Dict[str, Union[str, torch.Tensor]] = collections.OrderedDict()
        data_path = self.metadata[self.collection][id]
        for name, relative_path in data_path.items():
            full_path = os.path.join(self.task_root, relative_path)
            dtype = np.float32
            if name == 'label':
                dtype = np.long
            attribute_data = load_nifti(
                full_path,
                dtype=dtype,
                base_name=f'{name}_',
                remove_patient_transform=self.remove_patient_transform)
            data.update(**attribute_data)
        return data

    def __len__(self):
        return len(self.metadata[self.collection])


def _load_case_adaptor(batch: Batch, dataset: MedicalDecathlonDataset, transform_fn: Optional[Transform]):
    ids = batch['sample_uid']
    assert len(ids) == 1, 'only load a single case at a time!'
    data = dataset(ids[0])
    data['sample_uid'] = ids
    if transform_fn is not None:
        data = transform_fn(data)  # type: ignore

    return data


def create_decathlon_dataset(
        task_name: str,
        root: str = None,
        transform_train: Transform = None,
        transform_valid: Transform = None,
        nb_workers: int = 4,
        valid_ratio: float = 0.2,
        batch_size: int = 1,
        remove_patient_transform: bool = False) -> Datasets:
    """
    Create a task of the medical decathlon dataset.

    The dataset is available here http://medicaldecathlon.com/ with accompanying
    publication: https://arxiv.org/abs/1902.09063

    Args:
        task_name: the name of the task
        root: the root folder where the data will be created and possibly downloaded
        transform_train: a function that take a batch of training data and return a transformed batch
        transform_valid: a function that take a batch of valid data and return a transformed batch
        nb_workers: the number of workers used for the preprocessing
        valid_ratio: the ratio of validation data
        batch_size: the batch size
        remove_patient_transform: if ``True``, remove the affine transformation attached to the voxels

    Returns:
        a dictionary of datasets
    """
    root = get_data_root(root)

    dataset = MedicalDecathlonDataset(task_name, root, remove_patient_transform=remove_patient_transform)

    nb_samples = len(dataset)
    nb_train = int(nb_samples * (1.0 - valid_ratio))

    ids = np.arange(len(dataset))
    np.random.shuffle(ids)

    sampler_train = SamplerRandom(batch_size=batch_size)
    data_train = collections.OrderedDict([
        ('sample_uid', ids[:nb_train])
    ])
    load_data_train = functools.partial(_load_case_adaptor, dataset=dataset, transform_fn=transform_train)
    sequence_train = SequenceArray(data_train, sampler=sampler_train, sample_uid_name='sample_uid')
    sequence_train = sequence_train.map(
        function_to_run=load_data_train,
        nb_workers=nb_workers,
        max_jobs_at_once=4
    )
    sequence_train = sequence_train.collate()

    sampler_valid = SamplerSequential(batch_size=batch_size)
    data_valid = collections.OrderedDict([
        ('sample_uid', ids[nb_train:])
    ])

    load_data_valid = functools.partial(_load_case_adaptor, dataset=dataset, transform_fn=transform_valid)
    sequence_valid = SequenceArray(data_valid, sampler=sampler_valid, sample_uid_name='sample_uid')
    sequence_valid = sequence_valid.map(
        functools.partial(load_data_valid, dataset=dataset),
        nb_workers=nb_workers,
        max_jobs_at_once=4
    )
    sequence_valid = sequence_valid.collate()

    split = collections.OrderedDict([
        ('train', sequence_train),
        ('valid', sequence_valid),
    ])
    return {
        task_name: split
    }
