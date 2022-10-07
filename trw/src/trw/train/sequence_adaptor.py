from . import sequence
import collections
from torch.utils import data


class SequenceAdaptorTorch(sequence.Sequence, sequence.SequenceIterator):
    """
    Adapt a `torch.utils.data.DataLoader` to a `trw.train.Sequence` interface

    The main purpose is to enable compatibility with the torch data loader and any existing third party code.
    """
    def __init__(self, torch_dataloader, features=None):
        """
        
        Args:
            torch_dataloader: a `torch.utils.data.DataLoader` instance
            features: the name of the features returned by the dataloader. If `None`, use default numbered name
        """
        assert isinstance(torch_dataloader, data.DataLoader), 'must be a data loader'
        assert features is None or isinstance(features, collections.Sequence), 'must be a sequence'
        super().__init__(source_split=None)
        self.torch_dataloader = torch_dataloader
        self.iter_source = None
        self.features = features

    def __len__(self):
        return len(self.torch_dataloader)

    def __iter__(self):
        self.iter_source = self.torch_dataloader.__iter__()
        return self

    def __next__(self):
        batch = self.iter_source.__next__()
        if isinstance(batch, collections.Sequence):
            if self.features is None:
                self.features = ['feature_{}'.format(n) for n in range(len(batch))]
                
            # we want to name the element of the list
            batch_dict = collections.OrderedDict()
            for feature_index, feature_value in enumerate(batch):
                feature_name = self.features[feature_index]
                batch_dict[feature_name] = feature_value
            batch = batch_dict
        return batch

    def subsample(self, nb_samples):
        # we can't perform subsample, so return the original dataset
        return SequenceAdaptorTorch(torch_dataloader=self.torch_dataloader)

    def close(self):
        pass
