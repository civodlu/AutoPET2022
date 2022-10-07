import copy
from trw.basic_typing import Batch
import numpy as np
import torch


def transform_add_normalized_xyz_coordinates(batch: Batch, image_name: str = 'suv') -> Batch:
    """
    Create normalized XYZ coordinate maps from tiled input
    """
    new_batch = copy.copy(batch)

    current_shape_zyx = batch['current_shape'][::-1]
    block_shape = batch[image_name].shape
    chunking_offset_index_zyx = batch.get('chunking_offset_index_zyx')
    if chunking_offset_index_zyx is None:
        # if there is no chunking, it MUST be full size!
        assert (block_shape == batch['current_shape'][::-1]).all()
        chunking_offset_index_zyx = np.asarray([0, 0, 0])
    
    ratio_zyx_start = chunking_offset_index_zyx / current_shape_zyx
    coord_zyx_end = np.minimum((chunking_offset_index_zyx + block_shape), current_shape_zyx)
    ratio_zyx_end = coord_zyx_end / current_shape_zyx

    step = (ratio_zyx_end - ratio_zyx_start) / (np.asarray(block_shape))

    values = np.arange(ratio_zyx_start[0], ratio_zyx_end[0], step[0], dtype=np.float32)
    z_coords = np.tile(values.reshape((values.size, 1, 1)), [1, block_shape[1], block_shape[2]])
    values = np.arange(ratio_zyx_start[1], ratio_zyx_end[1], step[1], dtype=np.float32)
    y_coords = np.tile(values.reshape((1, values.size, 1)), [block_shape[0], 1, block_shape[2]])
    values = np.arange(ratio_zyx_start[2], ratio_zyx_end[2], step[2], dtype=np.float32)
    x_coords = np.tile(values.reshape((1, 1, values.size)), [block_shape[0], block_shape[1], 1])

    new_batch['z_coords'] = torch.from_numpy(z_coords)
    new_batch['y_coords'] = torch.from_numpy(y_coords)
    new_batch['x_coords'] = torch.from_numpy(x_coords)
    return new_batch