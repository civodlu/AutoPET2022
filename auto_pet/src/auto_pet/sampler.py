import collections
from trw.basic_typing import Batch
import numpy as np


def sample_tiled_volumes(
        case_data: Batch,
        *,
        tile_step=None,
        tile_size: int,
        random_offset=None,
        volume_name: str,
        tile_step_z=None) -> np.ndarray:
    """
    Tiles the whole volumes as a sub-volumes.

    Args:
        case_data: a single patient data
        random_offset: a random offset added to the tile. This is to make sure the sampling of the tile
            is truly random and not biased the patient position within the volume
        tile_size: the size of the file
        tile_step: the position offset between two tiles
    """
    samples = []
    if tile_step is None:
        tile_step = tile_size

    if tile_step_z is None:
        tile_step_z = tile_step

    shape = case_data[volume_name].shape
    assert len(shape) == 3

    if random_offset is not None and random_offset != 0:
        offset = np.random.random_integers(0, random_offset - 1, size=3)
    else:
        offset = [0, 0, 0]

    half_size = tile_size // 2
    for z in range(offset[0], shape[0], tile_step_z):
        for y in range(offset[1], shape[1], tile_step):
            for x in range(offset[2], shape[2], tile_step):
                center_voxel = np.asarray([z, y, x], dtype=int) + half_size
                samples.append(center_voxel)

    return np.asarray(samples)


def sample_random_subvolumes(
        case_data: Batch,
        *,
        tile_size: collections.Sequence,
        nb_samples: int,
        volume_name: str) -> np.ndarray:
    """
    Randomly sample positions within a 3D volume, considering margins (i.e., tile size)
    """
    shape = case_data[volume_name].shape
    assert len(shape) == 3

    if not isinstance(tile_size, collections.Sequence):
        half = [tile_size // 2] * 3
    else:
        assert len(tile_size) == 3

    offset_z = np.random.random_integers(0 + half[0], shape[0] - 1 - half[0], size=nb_samples)
    offset_y = np.random.random_integers(0 + half[1], shape[1] - 1 - half[1], size=nb_samples)
    offset_x = np.random.random_integers(0 + half[2], shape[2] - 1 - half[2], size=nb_samples)
    samples = np.asarray([offset_z, offset_y, offset_x]).transpose()
    return np.asarray(samples)
