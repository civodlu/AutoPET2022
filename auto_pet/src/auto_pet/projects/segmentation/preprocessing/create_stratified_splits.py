from collections import defaultdict
import copy
from glob import glob
import os
from typing import Dict, List, Optional, Sequence
from auto_pet.projects.segmentation.config import data
from auto_pet.projects.segmentation.preprocessing.preprocess_hdf5 import read_case_hdf5
from sklearn.model_selection import KFold
import numpy as np
import json


def make_split_descriptor(
        patients: Dict[str, List[str]],
        lesion_info: Dict,
        split: List[str],
        bin_edges: Sequence[int] = (1, 4, 8, 15)) -> np.ndarray:
    """
    Create a `descriptor` of the split: Average lesion volume, #lesions, lesion histogram
    """
    histogram_lesion_count = np.zeros(len(bin_edges) + 1)
    total_lesion_volumes = 0
    nb_scans = 0
    total_lesions = 0
    nb_no_lesion = 0
    for patient_name in split:
        segs = patients[patient_name]
        for s in segs:
            lesions = lesion_info[s]['segmentation_volumes']
            nb_lesions = len(lesions)
            lesion_volumes = sum(lesions)
            c = np.digitize(nb_lesions, bin_edges)
            histogram_lesion_count[c] += 1
            nb_scans += 1
            total_lesions += len(lesions)
            total_lesion_volumes += lesion_volumes
            if len(lesions) == 0:
                nb_no_lesion += 1

    average_lesion_volume = total_lesion_volumes / nb_scans
    average_lesion_volume_histogram = histogram_lesion_count / nb_scans
    average_lesion_per_case = total_lesions / nb_scans
    fraction_background_only = float(nb_no_lesion) / nb_scans
    return np.concatenate([
        np.asarray([
            average_lesion_volume / 50,  # just for normalization purposes
            average_lesion_per_case,
            fraction_background_only
        ]), 
        average_lesion_volume_histogram])


def eval_loss_splits(patients: Dict[str, List[str]], lesion_info: Dict, splits: List[List[str]]) -> float:
    descriptors = [make_split_descriptor(patients, lesion_info, split) for split in splits]
    nb_splits = len(splits)
    nb_pairs = 0
    accum_similarity = 0

    # we want the descriptors of each split to be as similar as possible
    for n1 in range(nb_splits):
        for n2 in range(n1 + 1, nb_splits):
            similarity = np.linalg.norm(descriptors[n1] - descriptors[n2])
            nb_pairs += 1
            accum_similarity += similarity
    
    return accum_similarity / nb_pairs


def create_splits(nb_folds: int, version: str, max_cases: Optional[int] = None, nb_test_cases: int = 75, nb_random_iter: int = 50000):
    """
    stratify the splits by avg lesion / avg lesion size / ratio of patients with lesions
    take into account the (possibly) multiple scans per patient 
    """
    patients = defaultdict(list)
    all_cases = glob(os.path.join(data.root_datasets, 'v1_3', '*.hdf5*'))
    if max_cases is not None:
        all_cases = all_cases[:max_cases]

    here = os.path.abspath(os.path.dirname(__file__))
    lesion_info_path = os.path.join(here, '..', 'config', 'cases_segmentation_info.json')
    with open(lesion_info_path, 'r') as f:
        lesion_info = json.load(f)

    # patients may have multiple timepoints!
    # so build a `patient` index
    print('Processing cases: ')
    for c_n, c in enumerate(all_cases):
        if c_n % 20 == 0:
            print('.', flush=True, end='')
        case_data = read_case_hdf5(c)
        patient_name = case_data['patient_name']
        
        # only keep the filename so that we can move the data
        # easily
        case_filepath = os.path.basename(c)
        patients[patient_name].append(case_filepath)

    # create the TEST split
    best_splits_loss = 1e10
    best_splits = None
    for _ in range(nb_random_iter):
        all_patient_names = np.asarray(list(patients.keys()))
        random_indices = np.arange(len(all_patient_names))
        np.random.shuffle(random_indices)
        test_patient_names = all_patient_names[random_indices[:nb_test_cases]]
        other_patient_names = all_patient_names[random_indices[nb_test_cases:]]
        assert len(set(test_patient_names).intersection(set(other_patient_names))) == 0
        
        splits = [other_patient_names, test_patient_names]
        splits_loss = eval_loss_splits(patients, lesion_info, splits)
        if splits_loss < best_splits_loss:
            best_splits_loss = splits_loss
            best_splits = splits
            print('best_splits_loss=', splits_loss)
    test_patient_names = best_splits[1]
    remaining_patient_names = best_splits[0]

    # use k-fold to find the train/valid splits
    best_splits_loss = 1e10
    best_splits = None
    kfold = KFold(n_splits=nb_folds)
    all_splits = list(kfold.split(remaining_patient_names))
    for _ in range(nb_random_iter):
        np.random.shuffle(remaining_patient_names)
        # only evaluate the valid splits, training is less important since
        # it is much larger
        splits = [remaining_patient_names[s[1]] for s in all_splits]
        splits_loss = eval_loss_splits(patients, lesion_info, splits)
        if splits_loss < best_splits_loss:
            best_splits_loss = splits_loss
            best_splits = copy.deepcopy(remaining_patient_names)
            print('best_splits_loss=', splits_loss)

    # export the splits
    for fold_n, (train_index, valid_index) in enumerate(all_splits):
        output_path = os.path.join(here, '..', 'config', f'{version}_kfold{fold_n}.json')
        assert not os.path.exists(output_path), f'split path={output_path} already exists!!'
        
        def collect_cases(selected_patients):
            # concatenate all the scans of all the patients
            paths = []
            for p in selected_patients:
                paths += patients[p]
            return paths

        train_cases = collect_cases(best_splits[train_index])
        valid_cases = collect_cases(best_splits[valid_index])
        test_cases = collect_cases(test_patient_names)
        assert len(set(test_cases).intersection(set(train_cases))) == 0
        assert len(set(test_cases).intersection(set(valid_cases))) == 0

        datasets = {
            'auto_pet': {
                'train': train_cases,
                'valid': valid_cases,
                'test': test_cases
            }
        }
        with open(output_path, 'w') as f:
            json.dump({'datasets': datasets}, f, indent=3)

if __name__ == '__main__':
    np.random.seed(0)
    #create_splits(nb_folds=3, version='autopet_stratified_v1_small', max_cases=50, nb_test_cases=15)
    create_splits(nb_folds=15, version='autopet_stratified_v1', nb_test_cases=75)
    print('DONE!')