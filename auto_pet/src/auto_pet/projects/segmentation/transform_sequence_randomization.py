from typing import Tuple
from trw.basic_typing import Batch
import numpy as np


def transform_sequence_randomize(
        batch: Batch, 
        sequence_features: Tuple[str, ...] = ('sequence_input', 'sequence_output', 'sequence_weight_volume'), subsample_probability: float=0, subsample_min_max_factors: Tuple[float, float]=(0.3, 0.8), min_lesions_after_subsampling: int = 3) -> Batch:
    """
    Randomize the sequence of segmentation descriptor to avoid overfitting and optionally subsample the sequence
    """
    apply_subsampling = np.random.rand() < subsample_probability

    size = batch[sequence_features[0]].shape[1]
    indices = np.arange(size)
    np.random.shuffle(indices)

    if apply_subsampling:
        subsample_ratio = np.random.uniform(subsample_min_max_factors[0], subsample_min_max_factors[1])
        nb_subsampled_segmentations = int(round(len(indices) * subsample_ratio))
        if nb_subsampled_segmentations >= min_lesions_after_subsampling:
            indices = indices[:nb_subsampled_segmentations]

    batch_new = {}
    for name, value in batch.items():
        if name in sequence_features:
            assert value.shape[1] == size, 'all features must have the SAME sequence size!'
            assert value.shape[0] == 1, 'TODO handle larger batches!'
            batch_new[name] = value[:, indices]
        else:
            batch_new[name] = value
    return batch_new