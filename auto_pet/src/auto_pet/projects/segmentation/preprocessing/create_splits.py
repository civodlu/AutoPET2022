from collections import defaultdict
from glob import glob
import os
from typing import Dict
from auto_pet.projects.segmentation.config import data
from auto_pet.projects.segmentation.preprocessing.preprocess import read_case
from sklearn.model_selection import KFold
import numpy as np
import json


def create_splits(nb_folds, version, max_cases=None):
    patients = defaultdict(list)
    all_cases = glob(os.path.join(data.root_datasets, 'v1_0', '*.pkl*'))
    if max_cases is not None:
        all_cases = all_cases[:max_cases]

    # patients may have multiple timepoints!
    # so build a `patient` index
    print('Processing cases: ')
    for c_n, c in enumerate(all_cases):
        if c_n % 20 == 0:
            print('.', flush=True, end='')
        case_data = read_case(c)
        patient_name = case_data['patient_name']
        
        # only keep the filename so that we can move the data
        # easily
        case_filepath = os.path.basename(c)
        patients[patient_name].append(case_filepath)


    here = os.path.abspath(os.path.dirname(__file__))
    all_patient_names = np.asarray(list(patients.keys()))
    kfold = KFold(n_splits=nb_folds)
    for fold_n, (train_index, test_index) in enumerate(kfold.split(all_patient_names)):
        output_path = os.path.join(here, '..', 'config', f'{version}_kfold{fold_n}.json')
        assert not os.path.exists(output_path), f'split path={output_path} already exists!!'
        
        def collect_cases(indices):
            paths = []
            for p in all_patient_names[indices]:
                paths += patients[p]
            return paths

        train_cases = collect_cases(train_index)
        test_cases = collect_cases(test_index)

        datasets = {
            'auto_pet': {
                'train': train_cases,
                'test': test_cases
            }
        }
        with open(output_path, 'w') as f:
            json.dump({'datasets': datasets}, f, indent=3)

if __name__ == '__main__':
    np.random.seed(0)
    #create_splits(nb_folds=10, version='autopet_v1')
    #create_splits(nb_folds=10, version='autopet_v1_small', max_cases=10)
    create_splits(nb_folds=10, version='autopet_v1_med', max_cases=50)
    print('DONE!')