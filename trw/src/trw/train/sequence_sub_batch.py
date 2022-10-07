from ..utils import len_batch
from . import sequence
from . import sequence_rebatch


class SequenceSubBatch(sequence.Sequence, sequence.SequenceIterator):
    """
    This sequence will split batches in smaller batches if the underlying sequence batch is too large.

    This sequence can be useful to manage very large tensors. Indeed, this class avoids
    concatenating tensors (as opposed to in :class:`trw.train.SequenceReBatch`). Since this operation
    can be costly as the tensors must be reallocated. In this case, it may be faster to
    work on a smaller batch by avoiding the concatenation cost.
    """
    def __init__(self, source_split, batch_size, discard_batch_not_full=False):
        """
        Args:
            source_split: the underlying sequence to normalize
            batch_size: the size of the batches created by this sequence
            discard_batch_not_full: if True, the last batch will be discarded if not full
        """
        super().__init__(source_split)

        assert batch_size > 0
        assert isinstance(source_split, sequence.Sequence), '`source_split` must be a `Sequence`'
        self.source_split = source_split
        self.batch_size = batch_size
        self.discard_batch_not_full = discard_batch_not_full
        self.iter_source = None
        self.iter_overflow = None

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)
        return SequenceSubBatch(
            subsampled_source,
            batch_size=self.batch_size,
            discard_batch_not_full=self.discard_batch_not_full)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)
        return SequenceSubBatch(
            subsampled_source,
            batch_size=self.batch_size,
            discard_batch_not_full=self.discard_batch_not_full)

    def __next__(self):

        nb_samples = 0
        batch = None
        try:
            if self.iter_overflow is not None and not self.discard_batch_not_full:
                # handles the samples that had previously overflown
                batch = self.iter_overflow
                self.iter_overflow = None
            else:
                # if not, get the next batch
                batch = self.iter_source.__next__()

            nb_samples = len_batch(batch)

            if nb_samples > self.batch_size:
                # too many samples, split the batch and keep the extra samples in the overflow
                batch, self.iter_overflow = sequence_rebatch.split_in_2_batches(batch, self.batch_size)

        except StopIteration:
            if nb_samples == 0:
                # end the sequence
                raise StopIteration()

        assert batch is not None, 'can\'t be ``None``, should have raised ``StopIteration``'
        return batch

    def __iter__(self):
        self.iter_source = self.source_split.__iter__()
        return self

    def close(self):
        if self.source_split is not None:
            self.source_split.close()

