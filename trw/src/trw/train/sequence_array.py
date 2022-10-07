from . import sequence
from . import sampler as sampler_trw
import numpy as np
import collections
import copy
from ..utils import get_batch_n, len_batch

# this the name used for the sample UID
sample_uid_name = 'sample_uid'


class SequenceArray(sequence.Sequence):
    """
    Create a sequence of batches from numpy arrays, lists and :class:`torch.Tensor`
    """
    def __init__(
            self,
            split,
            sampler=sampler_trw.SamplerRandom(),
            transforms=None,
            use_advanced_indexing=True,
            sample_uid_name=sample_uid_name):
        """

        Args:
            split: a dictionary of tensors. Tensors may be `numpy.ndarray`, `torch.Tensor`, numeric
            sampler: the sampler to be used to iterate through the sequence
            transforms: a transform or list of transforms to be applied on each batch of data
            use_advanced_indexing:
            sample_uid_name: if not `None`, create a unique UID per sample so that it is easy to track
                particular samples (e.g., during data augmentation)
        """
        super().__init__(None)  # there is no source sequence for this as we get our input from a numpy split
        self.split = split
        self.nb_samples = None
        self.sampler = sampler
        self.sampler_iterator = None
        self.transforms = transforms
        self.use_advanced_indexing = use_advanced_indexing

        # create a unique UID
        if sample_uid_name is not None and sample_uid_name not in split:
            split[sample_uid_name] = np.asarray(np.arange(len_batch(split)))

    def subsample(self, nb_samples):
        # get random indices
        subsample_sample = sampler_trw.SamplerRandom(batch_size=nb_samples)
        subsample_sample.initializer(self.split)

        # extract the indices
        indices = next(iter(subsample_sample))
        subsampled_split = get_batch_n(
            self.split,
            len_batch(self.split),
            indices,
            self.transforms,
            # use `use_advanced_indexing` so that we keep the types as close as possible to original
            use_advanced_indexing=True
        )
        return SequenceArray(
            subsampled_split,
            copy.deepcopy(self.sampler),
            transforms=self.transforms,
            use_advanced_indexing=self.use_advanced_indexing
        )

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        uid_values = self.split.get(uids_name)
        assert uid_values is not None, 'no UIDs with name={}'.format(uids_name)

        # find the samples that are in `uids`
        indices_to_keep = []
        uids_set = set(uids)
        for index, uid in enumerate(uid_values):
            if uid in uids_set:
                indices_to_keep.append(index)

        # reorder the `indices_to_keep` following the `uids` ordering
        uids_ordering = {uid: index for index, uid in enumerate(uids)}
        kvp_index_ordering = []
        for index in indices_to_keep:
            uid = uid_values[index]
            ordering = uids_ordering[uid]
            kvp_index_ordering.append((index, ordering))
        kvp_uids_ordering = sorted(kvp_index_ordering, key=lambda value: value[1])
        indices_to_keep = [index for index, ordering in kvp_uids_ordering]

        # extract the samples
        subsampled_split = get_batch_n(
            self.split,
            len_batch(self.split),
            indices_to_keep,
            self.transforms,
            # use `use_advanced_indexing` so that we keep the types as close as possible to original
            use_advanced_indexing=True
        )

        if new_sampler is None:
            new_sampler = copy.deepcopy(self.sampler)
        else:
            new_sampler = copy.deepcopy(new_sampler)

        return SequenceArray(
            subsampled_split,
            new_sampler,
            transforms=self.transforms,
            use_advanced_indexing=self.use_advanced_indexing
        )

    def __iter__(self):
        # make sure the sampler is copied so that we can have multiple iterators of the
        # same sequence
        return SequenceIteratorArray(self, copy.deepcopy(self.sampler))

    def close(self):
        pass


class SequenceIteratorArray(sequence.SequenceIterator):
    """
    Iterate the elements of an :class:`trw.train.SequenceArray` sequence

    Assumptions:
        - underlying `base_sequence` doesn't change sizes while iterating
    """
    def __init__(self, base_sequence, sampler):
        super().__init__()
        self.base_sequence = base_sequence
        self.nb_samples = len_batch(self.base_sequence.split)

        self.sampler = sampler
        self.sampler.initializer(self.base_sequence.split)
        self.sampler_iterator = iter(self.sampler)

    def __next__(self):
        indices = self.sampler_iterator.__next__()
        if not isinstance(indices, (np.ndarray, collections.Sequence)):
            indices = [indices]

        return get_batch_n(
            self.base_sequence.split,
            self.nb_samples,
            indices,
            self.base_sequence.transforms,
            self.base_sequence.use_advanced_indexing)

    def close(self):
        pass
