from typing import Optional, Callable

import numpy as np
import functools

from ..basic_typing import ShapeX, Datasets
from .dataset_fake_symbols import _random_color, _add_shape, _noisy, create_fake_symbols_datasset, ShapeCreator


def _add_square_3d(imag, mask, shapes_added, scale_factor):
    color = _random_color()

    shape = np.zeros([10, 10, 10], dtype=np.float)
    shape[1:9, 1:9, 1:9] = 1

    return _add_shape(imag, mask, shape, shapes_added, scale_factor, color)


def _add_rectangle_3d(imag, mask, shapes_added, scale_factor):
    color = _random_color()

    shape = np.zeros([10, 10, 10], dtype=np.float)
    shape[1:9, 3:7, 1:9] = 1

    return _add_shape(imag, mask, shape, shapes_added, scale_factor, color)


def default_shapes_3d(global_scale_factor=1.0):
    return [
        ('square_3d', functools.partial(_add_square_3d, scale_factor=global_scale_factor)),
        ('rectangle_3d', functools.partial(_add_rectangle_3d, scale_factor=global_scale_factor)),
    ]


def create_fake_symbols_3d_dataset(
        nb_samples: int,
        image_shape: ShapeX,
        ratio_valid: float = 0.2,
        nb_classes_at_once: Optional[int] = None,
        global_scale_factor: float = 1.0,
        normalize_0_1: bool = True,
        noise_fn: Callable[[np.ndarray], np.ndarray] = functools.partial(_noisy, noise_type='poisson'),
        shapes_fn: ShapeCreator = default_shapes_3d,
        max_classes: Optional[int] = None,
        batch_size: int = 64,
        background: int = 255,
        dataset_name: str = 'fake_symbols_3d') -> Datasets:
    """
    Create artificial 2D for classification and segmentation problems

    This dataset will randomly create shapes at random location & color with a segmentation map.

    Args:
        nb_samples: the number of samples to be generated
        image_shape: the shape of an image [height, width]
        ratio_valid: the ratio of samples to be used for the validation split
        nb_classes_at_once: the number of classes to be included in each sample. If `None`,
            all the classes will be included
        global_scale_factor: the scale of the shapes to generate
        noise_fn: a function to create noise in the image
        shapes_fn: the function to create the different shapes
        normalize_0_1: if True, the data will be normalized (i.e., image & position will be in range [0..1])
        max_classes: the total number of classes available
        batch_size: the size of the batch for the dataset
        background: the background value of the sample (before normalization if `normalize_0_1` is `True`)
        dataset_name: the name of the returned dataset

    Returns:
        a dict containing the dataset `fake_symbols_2d` with `train` and `valid` splits with features `image`,
        `mask`, `classification`, `<shape_name>_center`
    """
    assert len(image_shape) == 3, 'must be a DxHxW list'
    return create_fake_symbols_datasset(
        nb_samples=nb_samples,
        image_shape=image_shape,
        ratio_valid=ratio_valid,
        nb_classes_at_once=nb_classes_at_once,
        global_scale_factor=global_scale_factor,
        normalize_0_1=normalize_0_1,
        noise_fn=noise_fn,
        shapes_fn=shapes_fn,
        max_classes=max_classes,
        batch_size=batch_size,
        background=background,
        dataset_name=dataset_name
    )
