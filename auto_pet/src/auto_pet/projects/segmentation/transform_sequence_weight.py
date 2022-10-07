from trw.basic_typing import Batch
import numpy as np
import torch


def transform_sequence_randomize_create_weighting(batch: Batch, weight_volume_name: str = 'sequence_weight_volume', max_volume: float = 20000) -> Batch:
    """
    Create a weighting attributes for each element of a sequence.

    Here we want to heavily penalize large errors (rather than many small errors) and favor sensitivity over specificity  

    BEWARE: `sequence_input`, `sequence_output`, `sequence_descriptors` indices MUST match (e.g., randomization, subsampling)
    """
    sequence_descriptors = batch['sequence_descriptors']
    sequence_input = batch['sequence_input']
    sequence_output = batch['sequence_output']
    assert len(sequence_input) == 1, 'expected single case sequence!'
    assert len(sequence_output) == 1, 'expected single case sequence!'
    assert len(sequence_descriptors) == 1, 'expected single case sequence!'
    sequence_descriptors = sequence_descriptors[0]
    sequence_input = sequence_input[0]
    sequence_output = sequence_output[0]
    

    nb_elements = len(sequence_input)
    assert len(sequence_descriptors) == nb_elements
    assert len(sequence_descriptors) == len(sequence_output)

    w = torch.zeros(nb_elements, dtype=torch.float32, device=sequence_output.device)
    for n in range(nb_elements):
        w[n] = min(sequence_descriptors[n]['segmentation_volume'], max_volume) / max_volume
    batch[weight_volume_name] = w.unsqueeze(0)
    return batch