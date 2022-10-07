
import os
from trw.basic_typing import Batch
import time
from typing import Callable, Dict, Sequence, Tuple
from corelib import resample_sitk_image, make_sitk_image, get_sitk_image_attributes, make_sitk_image_attributes
import numpy as np
from SimpleITK import GetArrayFromImage
import torch
from glob import glob
import h5py
import cc3d
from trw.basic_typing import TorchTensorX
from trw.utils import sub_tensor
import lz4.frame
import pickle


def threshold_probability_map(probability_map: TorchTensorX, threshold: float=0.5) -> TorchTensorX:
    return probability_map >= threshold


def binary_dice(seg1: torch.tensor, seg2: torch.tensor) -> float:
    intersection = seg1 * seg2
    dice = 2 * intersection.sum() / (seg1.sum() + seg2.sum() + 1e-3)
    return dice


def describe_lesion(lesion_mask, suv, features, voxel_volume, wholebody_segmentation):
    assert len(wholebody_segmentation.shape) == 3
    voxel_n = np.where(lesion_mask)
    lesion_suv_values = suv[voxel_n]

    # average position (center of mass)
    segmentation_position = np.mean(voxel_n, axis=1) / lesion_mask.shape
    bb_min = np.min(voxel_n, axis=1)
    bb_max = np.max(voxel_n, axis=1)
    wholebody_segmentation_sub = sub_tensor(wholebody_segmentation, bb_min, bb_max + 1)
    lesion_mask_sub = sub_tensor(torch.from_numpy(lesion_mask), bb_min, bb_max + 1)
    dice = binary_dice(wholebody_segmentation_sub, lesion_mask_sub)

    # describe the shape of the lesion
    if len(voxel_n[0]) == 1:
        # only one voxel, cant calculate the STD
        segmentation_shape = np.array((0.01, 0.01, 0.01))
        suv_std = 0.1
    else:
        segmentation_shape = np.std(voxel_n, axis=1) / lesion_mask.shape * 100
        suv_std = lesion_suv_values.std()

    segmentation_volume = len(voxel_n[0]) * voxel_volume

    suv_mean = lesion_suv_values.mean()
    suv_max = lesion_suv_values.max()
    

    mean_feature = features.numpy().transpose((1, 2, 3, 0))[voxel_n]
    mean_feature = mean_feature.mean(axis=0)
    assert mean_feature.shape == (features.shape[0],)

    return {
        'segmentation_position': torch.from_numpy(segmentation_position),
        'segmentation_shape': torch.from_numpy(segmentation_shape),
        'segmentation_volume': torch.tensor(segmentation_volume),
        'suv_mean': torch.tensor(suv_mean),
        'suv_max': torch.tensor(suv_max),
        'suv_std': torch.tensor(suv_std),
        'mean_feature': torch.from_numpy(mean_feature),
        'segmentation_dice_with_truth': torch.tensor(dice)
    }


def create_segmentation_features(
        segmentation_descriptors: Batch, 
        min_dice_overlap: float = 0.5,
        feature_names: Sequence[str] = ('mean_feature', 'suv_mean', 'suv_max', 'suv_std', 'segmentation_volume', 'segmentation_shape', 'segmentation_position')) -> Tuple[torch.Tensor, torch.Tensor]:
    
    def cat_features(sequence_element):
        values = []
        for name in feature_names:
            v = sequence_element[name]
            if name in ('segmentation_volume',):
                v = torch.log(v + 1)

            v = v.reshape((1, v.numel()))
            values.append(v)

        return torch.cat(values, dim=1)
            
    segmentation_features = torch.cat([cat_features(d) for d in segmentation_descriptors], dim=0).type(torch.float32)
    segmentation_classification = torch.tensor([(d['segmentation_dice_with_truth'] >= min_dice_overlap).type(torch.int64) for d in segmentation_descriptors])
    
    # (N, SequenceSize, X) format
    return segmentation_features.unsqueeze(0), segmentation_classification.unsqueeze(0)


class TransformBuildLesionCandidates:
    """
    From a segmentation probability & feature map, build a sequence
    of lesion candidates to be post-processed by an RNN model

    Decouple lesion descriptors from sequence input so that the
    long calculations can be avoided for small input tweaks.
    """
    def __init__(
            self,
            precomputed_root: str,
            caching_root: str,
            data_version: str,
            feature_name: str = 'wholebody_inference_features',
            segmentation_prob_name: str = 'wholebody_inference',
            probability_to_segmentation_fn: Callable[[TorchTensorX], TorchTensorX] = threshold_probability_map,
            describe_lesion_fn = describe_lesion,
            create_segmentation_features_fn = create_segmentation_features,
            ) -> None:
        self.precomputed_root = precomputed_root
        self.feature_name = feature_name
        self.segmentation_prob_name = segmentation_prob_name
        self.probability_to_segmentation_fn = probability_to_segmentation_fn
        self.describe_lesion_fn = describe_lesion_fn
        self.create_segmentation_features_fn = create_segmentation_features_fn
        self.caching_root = caching_root
        self.data_version = data_version
        os.makedirs(caching_root, exist_ok=True)

    def calculate_sequence_features(self, batch: Batch) -> Batch:
        case_name = batch['case_name']

        time_start = time.perf_counter()

        for folder in (self.segmentation_prob_name, self.feature_name):
            matches = glob(os.path.join(self.precomputed_root, folder, f'*{case_name}output_raw.hdf5'))
            assert len(matches) == 1, f'got={matches}'
            with h5py.File(matches[0], 'r') as f:
                name = 'v'
                v = f[name][()]
                # always use torch as sharing mechanism
                batch[f'{folder}'] = torch.from_numpy(v)

        current_spacing = batch['current_spacing']
        voxel_volume = np.prod(current_spacing)
        features = batch[self.feature_name]
        segmentation_pb = batch[self.segmentation_prob_name]
        assert segmentation_pb.max() <= 1.0, f'must be probability! Got={segmentation_pb.max()}'

        segmentation_mask = self.probability_to_segmentation_fn(segmentation_pb)

        if isinstance(segmentation_mask, torch.Tensor):
            segmentation_mask = segmentation_mask.numpy()
        labels_out = cc3d.connected_components(segmentation_mask, connectivity=18)

        if labels_out.max() == 0:
            # discard this case, no segmentation found!
            # TODO should we use the truth to populate missing
            #   segmentation?
            return None

        segmentation_descriptors = []
        suv = batch['suv']
        for n in range(1, labels_out.max() + 1):
            lesion_mask = labels_out == n
            segmentation_descriptor = self.describe_lesion_fn(lesion_mask, suv, features, voxel_volume, batch['seg'])
            segmentation_descriptor['segmentation_id'] = n
            segmentation_descriptors.append(segmentation_descriptor)
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/lesion.npy', lesion_mask.astype(np.float32))
        
        new_batch = {'uid': [batch['uid']]}
        features_to_copy = ('case_name', 'timepoint', 'patient_name', 'current_spacing', 'current_origin', 'current_shape', 'mtv', 'suv', 'seg')
        for f_name in features_to_copy:
            new_batch[f_name] = batch[f_name]

        new_batch['sequence_label'] = torch.from_numpy(labels_out.astype(np.int16))
        new_batch['sequence_descriptors'] = segmentation_descriptors
        
        time_end = time.perf_counter()
        print(f'TransformBuildLesionCandidates.processing={time_end - time_start}')
        return new_batch

    def __call__(self, batch: Batch) -> Batch:
        case_name = batch['case_name']
        cache_name = os.path.join(self.caching_root, case_name + '.pkl')

        def recalculate_features(batch):
            batch = self.calculate_sequence_features(batch)
            with lz4.frame.open(cache_name, mode='wb') as f:
                pickle.dump({
                    'batch': batch,
                    'data_version': self.data_version,
                    'precomputed_root': self.precomputed_root
                }, f)
            return batch

        if not os.path.exists(cache_name):
            batch_sequence = recalculate_features(batch)
        else:
            with lz4.frame.open(cache_name, mode='rb') as f:
                data = pickle.load(f)

            if data['data_version'] != self.data_version or data['precomputed_root'] != self.precomputed_root:
                batch_sequence = recalculate_features(batch)
            else:
                batch_sequence = data['batch']

        if batch_sequence is None:
            return None

        sequence_input, sequence_output = self.create_segmentation_features_fn(batch_sequence['sequence_descriptors'])
        batch_sequence['sequence_input'] = sequence_input
        batch_sequence['sequence_output'] = sequence_output
        return batch_sequence

