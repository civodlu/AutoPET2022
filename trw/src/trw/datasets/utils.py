import os
from typing import Optional
import numpy as np
import torch
from PIL.Image import Image
from PIL import Image
import torchvision
from ..train import get_logging_root


def pic_to_tensor(pic: Image) -> torch.Tensor:
    assert isinstance(pic, Image), 'image must be a PIL Image'

    i = np.array(pic)
    if len(i.shape) == 2:
        i = np.reshape(i, [i.shape[0], i.shape[1], 1])
    i = i.transpose((2, 0, 1))
    return torch.from_numpy(i)


def pic_to_numpy(pic: Image) -> np.ndarray:
    assert isinstance(pic, Image), 'image must be a PIL Image'

    i = np.array(pic)
    if len(i.shape) == 2:
        i = np.reshape(i, [i.shape[0], i.shape[1], 1])
    i = i.transpose((2, 0, 1))
    return i


def get_data_root(data_root: Optional[str]) -> str:
    """
    Returns the location where all the data will be stored.

    data_root: a path where to store the data. if `data_root` is None,
        the environment variable `TRW_DATA_ROOT` will be used.
        If it is not defined, the default location of `get_logging_root` 
        will be used instead.
    """
    if data_root is None:
        # first, check if we have some environment variables configured
        data_root = os.environ.get('TRW_DATA_ROOT')

    if data_root is None:
        # else default a standard folder
        logging_root = get_logging_root(None)
        data_root = os.path.join(logging_root, 'datasets')

    assert data_root is not None

    data_root = os.path.expandvars(os.path.expanduser(data_root))
    return data_root


def download_and_extract_archive(url: str, dataset_path: str) -> None:
    from packaging.version import Version
    current_version = Version(torchvision.__version__)

    min_version = Version('0.2')
    if current_version < Version('0.2'):
        raise NotImplementedError(f'Can\'t download dataset with torchvision <= {min_version}')

    from torchvision.datasets.utils import download_and_extract_archive
    download_and_extract_archive(url, dataset_path)
