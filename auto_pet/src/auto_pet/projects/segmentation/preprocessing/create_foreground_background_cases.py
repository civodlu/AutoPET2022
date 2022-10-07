"""
Utility script to split the cases by foreground/no foreground.

The idea is to use this so that we have a 50/50% split
"""
from glob import glob
from auto_pet.projects.segmentation.config import data
import os
from auto_pet.projects.segmentation.preprocessing.preprocess import read_case
import json


all_cases = glob(os.path.join(data.root_datasets, 'v2_0', '*.pkl*'))

with_foreground = []
without_foreground = []
for c_n, c in enumerate(all_cases):
    print(f'Reading: {c_n + 1}/{len(all_cases)}')
    case_data = read_case(c)
    case_name = os.path.basename(c)

    if case_data['mtv'] > 0:
        with_foreground.append(case_name)
    else:
        without_foreground.append(case_name)

foreground_noforeground = {
    'with_foreground': with_foreground,
    'without_foreground': without_foreground
}

here = os.path.abspath(os.path.dirname(__file__))
output_path = os.path.join(here, '..', 'config', 'foreground_noforeground.json')
with open(output_path, 'w') as f:
    json.dump(foreground_noforeground, f, indent=3)