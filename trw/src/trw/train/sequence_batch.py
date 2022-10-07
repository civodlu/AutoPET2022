from . import sequence


class SequenceBatch(sequence.Sequence, sequence.SequenceIterator):
    """
    Group several batches into a single batch
    """
    def __init__(self, source_split, batch_size, discard_batch_not_full=False, collate_fn=sequence.default_collate_list_of_dicts):
        """
        Group the samples into a batch.

        :param batch_size: the number of samples used to create a new batch of data
        :param discard_batch_not_full: if True and if a batch is not full, discard it
        :param collate_fn: the function to assemble a list of items. If None, return the items as they were in `source_split`
        """
        super().__init__(source_split)

        assert batch_size > 0
        assert isinstance(source_split, sequence.Sequence), '`source_split` must be a `Sequence`'
        self.source_split = source_split
        self.batch_size = batch_size
        self.discard_batch_not_full = discard_batch_not_full
        self.iter_source = None
        self.collate_fn = collate_fn

    def subsample(self, nb_samples):
        subsampled_source = self.source_split.subsample(nb_samples)
        return SequenceBatch(subsampled_source, batch_size=self.batch_size, discard_batch_not_full=self.discard_batch_not_full, collate_fn=self.collate_fn)

    def subsample_uids(self, uids, uids_name, new_sampler=None):
        subsampled_source = self.source_split.subsample_uids(uids, uids_name, new_sampler)
        return SequenceBatch(subsampled_source, batch_size=self.batch_size, discard_batch_not_full=self.discard_batch_not_full, collate_fn=self.collate_fn)

    def __next__(self):
        items = []
        try:
            for i in range(self.batch_size):
                item = self.iter_source.__next__()
                if isinstance(item, list):
                    # multiple items, concatenate all the items at once
                    items += item
                else:
                    items.append(item)
        except StopIteration:
            if len(items) == 0 or (len(items) != self.batch_size and self.discard_batch_not_full):
                raise StopIteration()
        if self.collate_fn is not None:
            return self.collate_fn(items)
        return items

    def __iter__(self):
        self.iter_source = self.source_split.__iter__()
        return self

    def close(self):
        if self.source_split is not None:
            self.source_split.close()
