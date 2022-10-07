from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import functools
import skimage
import collections
import trw
import torch
from ..basic_typing import ShapeX, Datasets
from typing_extensions import Protocol, Literal


def _noisy(image: np.ndarray, noise_type: Literal['gauss', 'poisson', 's&p', 'speckle']) -> np.ndarray:
    """

    Args:
        image: a numpy image (float) in range [0..255]
        noise_type: the type of noise. Must be one of:

        * 'gauss'     Gaussian-distributed additive noise.
        * 'poisson'   Poisson-distributed noise generated from the data.
        * 's&p'       Replaces random pixels with 0 or 1.
        * 'speckle'   Multiplicative noise using out = image + n*image, where n is
            uniform noise with specified mean & variance

    Returns:
        noisy image
    """
    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss * 0.5
        return noisy

    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out

    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_type == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

    raise NotImplemented()


def _random_location(image_shape: np.ndarray, figure_shape) -> np.ndarray:
    image_shape = np.asarray(image_shape)
    figure_shape = np.asarray(figure_shape)
    maximum_location = image_shape - figure_shape
    location = [np.random.randint(0, l + 1) for l in maximum_location]
    return np.asarray(location)


def _random_color() -> np.ndarray:
    color = np.random.randint(0, 255 + 1, [3]).astype(np.uint8).reshape([3, 1])
    return color


def _add_shape(imag, mask, shape, shapes_added, scale_factor, color, min_overlap_distance=30):
    shape_shape = np.asarray(shape.shape)
    shape_imag_2d = imag.shape[1:]
    expected_shape = (np.asarray(shape_shape) * scale_factor).astype(int) + 1

    distance = 0
    location = None
    while distance <= min_overlap_distance:
        distance = 1e8
        location = _random_location(shape_imag_2d, expected_shape)
        center = location + shape_shape / 2

        for location_min, location_max in shapes_added:
            center_shape = (location_min + location_max) / 2
            distance_shape = np.linalg.norm(center_shape - center)
            if distance_shape < distance:
                distance = distance_shape

    shape_scaled = skimage.transform.rescale(shape, scale_factor, order=0, multichannel=False)
    indices = np.where(shape_scaled > 0)
    indices = [
        coords + location[index] for index, coords in enumerate(indices)
    ]

    # TODO generalize to N > 3
    if len(indices) == 2:
        imag[:, indices[0], indices[1]] = color
        mask[indices[0], indices[1]] = 1
    elif len(indices) == 3:
        imag[:, indices[0], indices[1], indices[2]] = color
        mask[indices[0], indices[1], indices[2]] = 1
    else:
        raise NotImplemented()
    return location, location + np.asarray(shape_scaled.shape)


def _create_image(shape, objects, nb_classes_at_once=None, max_classes=None, background=255):
    """

    Args:
        shape: the shape of an image [height, width]
        nb_classes_at_once: the number of classes to be included in each sample. If `None`,
            all the classes will be included
        max_classes: the maximum number of classes to be used. If `None`, all classes can
            be used, else a random subset

    Returns:
        image, mask and shape information
    """

    if max_classes is not None:
        objects = objects[:max_classes]

    if nb_classes_at_once is not None:
        np.random.shuffle(objects)
        objects = objects[:nb_classes_at_once]

    img = np.zeros([3] + list(shape), dtype=np.uint8)
    img.fill(background)
    mask = np.zeros(shape, dtype=np.int64)

    shapes_added = []
    shapes_infos = []
    for fn_name, fn in objects:
        shape = fn(img, mask, shapes_added)
        shapes_added.append(shape)
        shapes_infos.append((fn_name, ((shape[0] + shape[1]) / 2).astype(np.float32)))

    return img, mask, shapes_infos


class ShapeCreator(Protocol):
    def __call__(self, global_scale_factor: float) -> \
            Dict[str, Callable[[Any], Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]]]: ...


def create_fake_symbols_datasset(
        nb_samples: int,
        image_shape: ShapeX,
        dataset_name: str,
        shapes_fn: ShapeCreator,
        ratio_valid: float = 0.2,
        nb_classes_at_once: Optional[int] = None,
        global_scale_factor: float = 1.0,
        normalize_0_1: bool = True,
        noise_fn: Callable[[np.ndarray], np.ndarray] = functools.partial(_noisy, noise_type='poisson'),
        max_classes: Optional[int] = None,
        batch_size: int = 64,
        background: int = 255) -> Datasets:
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
    class_dict: Dict = collections.OrderedDict()
    samples = []

    objects = shapes_fn(global_scale_factor=global_scale_factor)

    for i in range(nb_samples):
        r = _create_image(image_shape, nb_classes_at_once=nb_classes_at_once, objects=objects, max_classes=max_classes, background=background)
        samples.append(r)

        for class_name, _ in r[2]:
            if class_name not in class_dict:
                class_dict[class_name] = len(class_dict)

    # unpack the shape infos so that we have the same size for all features
    dataset = collections.defaultdict(list)
    for image, mask, shapes in samples:
        if noise_fn is not None:
            image = noise_fn(image)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if normalize_0_1:
            image = (image.astype(np.float32) / 255.0 - 0.7) * 5
        dataset['image'].append(image)
        dataset['mask'].append(mask.reshape([1] + list(mask.shape)))

        shapes_dict = dict(shapes)

        if nb_classes_at_once == 1:
            # if we have only one shape per sample, group all classes in
            # a single feature
            c = next(iter(shapes_dict.keys()))
            c_id = class_dict[c]
            dataset['classification'].append(c_id)

        for class_name, class_id in class_dict.items():
            shape_pos = shapes_dict.get(class_name)
            if shape_pos is not None:
                if normalize_0_1:
                    shape_pos /= np.asarray(image_shape, dtype=np.float32)
                dataset[class_name].append(1)
                dataset[class_name + '_center'].append(shape_pos)
            else:
                dataset[class_name].append(0)
                dataset[class_name + '_center'].append(np.zeros(dtype=np.float32, shape=[len(image_shape)]))

    nb_valid = int(ratio_valid * nb_samples)
    nb_train = nb_samples - nb_valid

    dataset_train = {}
    dataset_valid = {}
    for feature_name, feature_values in dataset.items():
        if feature_name in class_dict or feature_name == 'classification':
            feature_values = np.asarray(feature_values, dtype=np.int64)
        dataset_train[feature_name] = torch.from_numpy(np.asarray(feature_values[:nb_train]))
        dataset_valid[feature_name] = torch.from_numpy(np.asarray(feature_values[nb_train:]))

    return {
        dataset_name: {
            'train': trw.train.SequenceArray(dataset_train, sampler=trw.train.SamplerRandom(batch_size=batch_size)),
            'valid': trw.train.SequenceArray(dataset_valid, sampler=trw.train.SamplerRandom(batch_size=batch_size)),
        }
    }



