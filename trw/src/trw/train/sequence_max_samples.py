from ..utils import len_batch
from . import sequence


class SequenceMaxSamples(sequence.Sequence, sequence.SequenceIterator):
    """
    Virtual resize of the sequence. The sequence will terminate when a certain number
        of samples produced has been reached. Restart the sequence where it was stopped.
    """
    def __init__(self, source_split, max_samples):
        """
        Normalize a sequence to identical batch size given an input sequence with varying batch size

        Args:
            source_split: the underlying sequence to normalize
            max_samples: the number of samples this sequence will produce before stopping
        """
        super().__init__(source_split)

        assert max_samples > 0
        assert isinstance(source_split, sequence.Sequence), '`source_split` must be a `Sequence`'
        self.max_samples = max_samples
        self.source_split = source_split
        self.iter_source = None
        self.current_nb_samples = None

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)
        return SequenceMaxSamples(subsampled_source, max_samples=self.max_samples)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)
        return SequenceMaxSamples(subsampled_source, max_samples=self.max_samples)

    def __next__(self):
        while True:
            if self.current_nb_samples >= self.max_samples:
                raise StopIteration()

            try:
                # if not, get the next batch
                batch = self.iter_source.__next__()
            except StopIteration:
                self.iter_source = self.source_split.__iter__()
                batch = self.iter_source.__next__()

            if batch is None or len(batch) == 0:
                # for some reason, the batch is empty
                # get a new one!
                continue

            self.current_nb_samples += len_batch(batch)
            return batch

    def __iter__(self):
        # restart the sequence count
        self.current_nb_samples = 0

        # do NOT reinitialize the underlying source: we want to continue
        # where we stopped
        if self.iter_source is None:
            # first time this sequence is called: start the underlying sequence if it was.
            self.iter_source = self.source_split.__iter__()
        return self

    def close(self):
        if self.source_split is not None:
            self.source_split.close()
