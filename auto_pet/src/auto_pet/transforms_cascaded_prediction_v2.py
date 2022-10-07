
import os
from trw.basic_typing import Batch
import time
from typing import Callable, Dict
from corelib import resample_sitk_image, make_sitk_image, get_sitk_image_attributes, make_sitk_image_attributes
import numpy as np
from SimpleITK import GetArrayFromImage
import torch
from glob import glob
import h5py
import copy


def get_case_files_hdf5(root, case_name, target_origin_xyz, target_spacing_xyz, target_shape_xyz, batch: Batch) -> None:
    # does not handle chunking so `chunking_origin` and `chunking_shape` are not relevant
    matches = glob(os.path.join(root, f'*{case_name}output_raw.hdf5'))
    assert len(matches) == 1, f'got={matches}'
    
    case_data = {}
    time_loading_start = time.perf_counter()
    with h5py.File(matches[0], 'r') as f:
        preprocessed_spacing = copy.deepcopy(f['spacing'][()])
        preprocessed_origin = copy.deepcopy(f['origin'][()])

        target_index_in_chunk_xyz = np.floor((target_origin_xyz - preprocessed_origin) / preprocessed_spacing).astype(int)
        target_shape_in_chunk_xyz = np.ceil((target_spacing_xyz * target_shape_xyz) / preprocessed_spacing).astype(int)
        # recalculate the origin of the index after rounding
        chunk_origin = target_index_in_chunk_xyz * preprocessed_spacing + preprocessed_origin

        name = 'v'
        v = copy.deepcopy(f[name][
            target_index_in_chunk_xyz[2]:target_index_in_chunk_xyz[2]+target_shape_in_chunk_xyz[2]+1,
            target_index_in_chunk_xyz[1]:target_index_in_chunk_xyz[1]+target_shape_in_chunk_xyz[1]+1,
            target_index_in_chunk_xyz[0]:target_index_in_chunk_xyz[0]+target_shape_in_chunk_xyz[0]+1,
        ])
        case_data['cascade.inference.output_found'] = v
        #print('MAX=', v.max())
    time_loading_end = time.perf_counter()

    time_resampling_start = time.perf_counter()
    for name, image in case_data.items():
        # resample the data tp original spacing
        image_sitk = make_sitk_image(
            image, 
            origin_xyz=chunk_origin, 
            spacing_xyz=preprocessed_spacing
        )
        fill_value = image.min()

        # remember SITK operates in XYZ not ZYX
        target_attributes = get_sitk_image_attributes(image_sitk)
        target_attributes['spacing'] = target_spacing_xyz
        target_attributes['shape'] = target_shape_xyz
        target_attributes['origin'] = target_origin_xyz

        image_resampled_itk = resample_sitk_image(image_sitk, attributes=target_attributes, fill_value=float(fill_value))
        image_resampled_npy = GetArrayFromImage(image_resampled_itk)
        assert (image_resampled_npy.shape == target_shape_xyz[::-1]).all()
        batch[name] = torch.from_numpy(image_resampled_npy)

        # debug
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp2/suv.npy', batch['suv'].numpy())
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp2/v.npy', v)
        #np.save('/mnt/datasets/ludovic/AutoPET/tmp2/cascade.inference.output_found.npy', batch['cascade.inference.output_found'].numpy())
    
    time_resampling_end = time.perf_counter()
    #print(f'cascade time loading={time_loading_end - time_loading_start}, resampling={time_resampling_end - time_resampling_start}')


class TransformCascadedPredictionPrecomputedV2:
    """
    The cascading is already on the filesystem. Just reload and resample to current spacing
    """
    def __init__(
            self,
            precomputed_root: str,
            case_name_to_files: Callable[[str, str], Dict] = get_case_files_hdf5,
            ) -> None:
        self.precomputed_root = precomputed_root
        self.case_name_to_files = case_name_to_files

    def __call__(self, batch: Batch) -> Batch:
        one_image_name = [n for n, v in batch.items() if isinstance(v, torch.Tensor) and len(v.shape) == 3]
        assert len(one_image_name) >= 1, 'no image found!'
        case_name = batch['case_name']
        target_shape_xyz = np.asarray(batch['ct'].shape)[::-1]
        target_origin_xyz = batch['current_origin']
        target_spacing_xyz = batch['current_spacing']

        self.case_name_to_files(
            self.precomputed_root, 
            case_name, 
            target_origin_xyz,
            target_spacing_xyz,
            target_shape_xyz,
            batch=batch
        )
        return batch

