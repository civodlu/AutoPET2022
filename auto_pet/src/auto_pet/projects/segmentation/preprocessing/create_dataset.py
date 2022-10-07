import copy
from functools import partial
import os
from typing import Dict, Optional, Tuple

from auto_pet.projects.segmentation.config import data
from auto_pet.projects.segmentation.preprocessing.preprocess import write_case as write_case_pkl
from auto_pet.projects.segmentation.preprocessing.preprocess_hdf5 import write_case_hdf5, read_case_hdf5, case_image_sampler_random
import glob
from corelib import read_nifti, get_sitk_image_attributes, resample_sitk_image
from SimpleITK import GetArrayFromImage, sitkNearestNeighbor, Image
import SimpleITK as sitk
import numpy as np


def create_case(uid: str, ct: Image, suv: Image, seg: Optional[Image], spacing: Tuple[float, float, float], case_name: Optional[str] = None, timepoint: Optional[str] = None, patient_name: Optional[str] = None, interpolator=sitk.sitkLinear) -> Dict:
    ct_np = GetArrayFromImage(ct)
    suv_np = GetArrayFromImage(suv)
    assert ct_np.shape == suv_np.shape, f'shape mismatch: {ct_np.shape}, {suv_np.shape}'
    if seg is not None:
        seg_np = GetArrayFromImage(seg)
        assert ct_np.shape == seg_np.shape

    attributes = get_sitk_image_attributes(suv)
    target_attributes = copy.copy(attributes)

    print(f'spacing={attributes["spacing"]}')

    # resampled to a specific spacing
    if spacing is not None:
        spacing_target = np.asarray(spacing, dtype=float)
        target_attributes['spacing'] = spacing_target
        target_attributes['shape'] = (np.asarray(attributes['shape']) * np.asarray(attributes['spacing']) / spacing_target).round().astype(np.int32)

        resampled_ct = resample_sitk_image(ct, target_attributes, fill_value=-1024.0, interpolator=interpolator)
        ct_np = GetArrayFromImage(resampled_ct)
        resampled_suv = resample_sitk_image(suv, target_attributes, fill_value=0, interpolator=interpolator)
        suv_np = GetArrayFromImage(resampled_suv)
        if seg is not None:
            resampled_seg = resample_sitk_image(seg, target_attributes, fill_value=0, interpolator=sitkNearestNeighbor) 
            seg_np = GetArrayFromImage(resampled_seg)
    else:
        # if we don't resample, the voxels MUST be aligned!
        assert ct_np.shape == suv_np.shape
        if seg is not None:
            assert ct_np.shape == seg_np.shape

    case_data = {
        'uid': uid,
        'patient_name': patient_name,
        'timepoint': timepoint,
        'ct': ct_np,
        'suv': suv_np,

        # we need this tag to do the inverse pre-processing
        'original_spacing': attributes['spacing'],
        'original_origin': attributes['origin'],
        'original_shape': attributes['shape'],
        'original_direction': attributes['direction'],
    }

    if seg is not None:
        case_data['seg'] = seg_np
        case_data['mtv'] = float(seg_np.sum())

    if spacing is not None:
        case_data['target_spacing'] = target_attributes['spacing']
        case_data['target_origin'] = target_attributes['origin']
        case_data['target_shape'] =  target_attributes['shape']

    return case_data


def create_dataset(version='v1_0', spacing=(6, 6, 6), interpolator=sitk.sitkLinear, write_case_fn=write_case_pkl):
    datasets_output = os.path.join(data.root_datasets, version)
    os.makedirs(datasets_output, exist_ok=True)

    cases = glob.glob(os.path.join(data.root_nifti, '*'))
    for c_n, c_root in enumerate(cases):
        timepoints = os.listdir(c_root)
        print(f'processing={c_root}, {c_n + 1}/{len(cases)}, TP={len(timepoints)}')
        for timepoint in timepoints:
            c = os.path.join(c_root, timepoint)
            ct_path = os.path.join(c, 'CTres.nii.gz')
            ct = read_nifti(ct_path)
            suv_path = os.path.join(c, 'SUV.nii.gz')
            suv = read_nifti(suv_path)
            seg_path = os.path.join(c, 'SEG.nii.gz')
            seg = read_nifti(seg_path)

            case_name = os.path.basename(c_root) + '_tp_' + timepoint[:15]
            patient_name = os.path.basename(c_root)
            case_data = create_case(
                uid=case_name, 
                ct=ct, 
                suv=suv, 
                seg=seg, 
                spacing=spacing, 
                case_name=case_name, 
                patient_name=patient_name, 
                timepoint=timepoint,
                interpolator=interpolator
            )
            # find all the bounding boxes locations, can be useful
            # for the sampling
            from auto_pet.projects.segmentation.transform_label_seg import transform_label_segmentation
            case_data = transform_label_segmentation(case_data)

            output_case_path = os.path.join(datasets_output, case_name + '.pkl.lz4')
            write_case_fn(output_case_path, case_data)

    print('Processing DONE!')



if __name__ == '__main__':
    # new HDF5 format
    create_dataset(version='v1_3', spacing=(6, 6, 6), write_case_fn=write_case_hdf5)
    
    # no resampling HDF5 for efficient sub-volume (chunking) loading & compression
    # We need to be able to load as many cases as possible to have large
    # data diversity during an epoch to get a robust estimate of the dice. Native
    # resolution makes this difficult... instead load part of the data only
    create_dataset(version='v4_0', spacing=None, write_case_fn=write_case_hdf5)