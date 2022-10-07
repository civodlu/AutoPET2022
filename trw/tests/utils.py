import torch
import torch.utils
import torch.utils.data
import numpy as np
import tempfile


# root_output = 'c:/tmp/'
root_output = tempfile.mkdtemp()

class NumpyDatasets(torch.utils.data.Dataset):
    """
    Simple numpy based arrays.

    All arrays must be named
    """
    def __init__(self, **kwargs):
        super(NumpyDatasets, self).__init__()
        self.features = kwargs

        self.size = None
        for feature_name, values in self.features.items():
            assert isinstance(values, np.ndarray), 'must be an array!'
            if self.size is None:
                self.size = len(values)
            assert self.size == len(values), 'all features must have the same size!'

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        sample = {}
        for feature_name, values in self.features.items():
            if len(values.shape) == 1:
                v = values[item:item + 1]  # keep the value as an array!
            else:
                v = values[item]
            sample[feature_name] = torch.Tensor(v)
        return sample
