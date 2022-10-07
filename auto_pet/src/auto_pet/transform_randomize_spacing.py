from typing import Callable, Sequence, Tuple
import numpy as np
import torch
from trw.transforms import affine_transform
from trw.transforms.transforms_affine import _random_affine_3d
from trw.basic_typing import Batch
import trw.transforms


def set_current_spacing_fn(batch: Batch, spacing: np.ndarray):
    batch['current_spacing'] = spacing

class TransformRandomizeSpacing:
    """
    Randomly resample the volumes. This can be useful to simulate
    arbitrary volume spacing.

    Volume shapes are expected to be `DHW` format.
    """
    def __init__(
            self, 
            p: float = 0.5, 
            xy_dependent: bool = True, 
            base_volume_name: str = 'suv',
            get_current_spacing_fn: Callable[[Batch], np.ndarray] = lambda batch: np.asarray(batch['current_spacing']),
            set_current_spacing_fn: Callable[[Batch, np.ndarray], None] = set_current_spacing_fn,
            spacing_min_xyz: Tuple[float, float, float] = (1.5, 1.5, 2.0),
            spacing_max_xyz: Tuple[float, float, float] = (4.0, 4.0, 4.0),) -> None:
        self.p = p
        self.xy_dependent = xy_dependent
        self.spacing_min_xyz = spacing_min_xyz
        self.spacing_max_xyz = spacing_max_xyz
        self.get_current_spacing_fn = get_current_spacing_fn
        self.set_current_spacing_fn = set_current_spacing_fn
        self.base_volume_name = base_volume_name

    def __call__(self, batch: Batch) -> Batch:
        if np.random.rand() > self.p:
            return batch

        target_spacing_xyz = [np.random.uniform(low, high) for low, high in zip(self.spacing_min_xyz, self.spacing_max_xyz)]
        if self.xy_dependent:
            target_spacing_xyz[1] = target_spacing_xyz[0]
        target_spacing_xyz = np.asarray(target_spacing_xyz)

        current_spacing_xyz = self.get_current_spacing_fn(batch)
        shape = batch[self.base_volume_name].shape
        target_shape = np.round(np.asarray(shape) * current_spacing_xyz[::-1] / target_spacing_xyz[::-1]).astype(int)
        assert len(shape) == 3, 'expected DHW format!'

        new_batch = {}
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                added_dim = 0
                if len(value.shape) == 3 and value.shape == shape:
                    added_dim = 2
                elif len(value.shape) == 4 and value.shape[1:] == shape:
                    # if we have a `C` component
                    added_dim = 1
                else:
                    new_batch[name] = value
                    continue
                for _ in range(added_dim):
                    value = value.unsqueeze(0)
                
                mode = 'linear'
                if value.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                    mode = 'nearest'
                value_resized = trw.transforms.resize(value, target_shape, mode=mode)
                for _ in range(added_dim):
                    value_resized = value_resized.squeeze(0)
                new_batch[name] = value_resized
            else:
                new_batch[name] = value

        # update the spacing so that the geometry info is up to date
        self.set_current_spacing_fn(new_batch, current_spacing_xyz)
        return new_batch
