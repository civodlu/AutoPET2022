from unittest import TestCase
import trw.train
import torch
from torch.utils import data
import collections


class DatasetDict(data.Dataset):
    def __init__(self, X, y):
        self.y = y
        self.X = X

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            'X': self.X[index],
            'y': self.y[index]
        }


class DatasetList(data.Dataset):
  def __init__(self, X, y):
        self.y = y
        self.X = X

  def __len__(self):
        return len(self.y)

  def __getitem__(self, index):
        return self.X[index], self.y[index]


class TestSequenceAdaptor(TestCase):
    def test_pytorch_adaptor_dict(self):
        dataset = DatasetDict(X=torch.ones([10, 2]), y=torch.zeros(10))
        data_loader = data.DataLoader(dataset, num_workers=1)

        batches = []
        for batch in trw.train.SequenceAdaptorTorch(data_loader):
            assert isinstance(batch, collections.Mapping), 'must be a dictionary of tensor features'
            batches.append(batch)
        self.assertTrue(len(batches) == 10)

    def test_pytorch_adaptor_list(self):
        dataset = DatasetList(X=torch.ones([10, 2]), y=torch.zeros(10))
        data_loader = data.DataLoader(dataset, num_workers=1)

        batches = []
        for batch in trw.train.SequenceAdaptorTorch(data_loader).batch(2):
            assert isinstance(batch, collections.Mapping), 'must be a dictionary of tensor features'
            batches.append(batch)
        self.assertTrue(len(batches) == 5)
