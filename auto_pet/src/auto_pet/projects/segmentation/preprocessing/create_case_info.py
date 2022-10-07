"""
Utility script to create useful statistics for each case.

This will be used to stratify the data into different data splits
"""

from glob import glob
from auto_pet.projects.segmentation.config import data
import os
from auto_pet.projects.segmentation.preprocessing.preprocess_hdf5 import read_case_hdf5
import json
import cc3d
import numpy as np


all_cases = glob(os.path.join(data.root_datasets, 'v1_3', '*.hdf5*'))

cases_info = {}
for c_n, c in enumerate(all_cases):
    print(f'Reading: {c_n + 1}/{len(all_cases)}')
    case_data = read_case_hdf5(c)
    case_name = os.path.basename(c)
    current_spacing = case_data['current_spacing']
    voxel_volume = np.prod(current_spacing) / 1000.0  # ml

    seg = case_data['seg']
    labels_out = cc3d.connected_components(seg.numpy(), connectivity=18)
    max_labels = labels_out.max()
    segmentation_volumes = []
    for n in range(1, max_labels):
        nb_voxels = (labels_out == n).sum()
        segmentation_volume = nb_voxels * voxel_volume
        segmentation_volumes.append(segmentation_volume)

    case_info = {'segmentation_volumes': segmentation_volumes}
    cases_info[case_data['case_name']] = case_info

here = os.path.abspath(os.path.dirname(__file__))
output_path = os.path.join(here, '..', 'config', 'cases_segmentation_info.json')
with open(output_path, 'w') as f:
    json.dump(cases_info, f, indent=3)