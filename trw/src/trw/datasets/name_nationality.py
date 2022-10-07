#
# Inspired from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
#

import collections
import glob
import os
import unicodedata
import string
from typing import Optional, Any

import torch
from ..train import SamplerRandom, SequenceArray
from ..basic_typing import Datasets
from .utils import download_and_extract_archive, get_data_root


def find_files(root):
    return glob.glob(os.path.join(root, 'data/names/*.txt'))


def unicode_to_ascii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_file(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line, all_letters) for line in lines]


def letter_to_index(letter, all_letters):
    return all_letters.find(letter)


def letter_to_tensor(letter, all_letters):
    tensor = torch.zeros(1, len(all_letters))
    tensor[0][letter_to_index(letter, all_letters)] = 1
    return tensor


def line_to_tensor(line, all_letters):
    """
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors
    """
    tensor = torch.zeros(len(line), 1, len(all_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter, all_letters)] = 1
    return tensor.unsqueeze(0)


def create_name_nationality_dataset(
        url: str = 'https://download.pytorch.org/tutorial/data.zip',
        root: Optional[str] = None,
        valid_ratio: float = 0.1,
        seed: int = 0,
        batch_size: int = 1) -> Datasets:
    torch.manual_seed(seed)
    root = get_data_root(root)

    dataset_path = os.path.join(root, 'name_nationality')
    download_and_extract_archive(url, dataset_path)

    # build the category_lines dictionary, a list of names per language
    all_letters = string.ascii_letters + " .,;'"
    category_lines = collections.OrderedDict()
    for filename in find_files(dataset_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        lines = read_file(filename, all_letters)
        category_lines[category] = lines

    valid_split: Any = collections.defaultdict(list)
    train_split: Any = collections.defaultdict(list)
    for category_id, (category, lines) in enumerate(category_lines.items()):
        lines_torch = [line_to_tensor(line, all_letters) for line in lines]

        indices = torch.randperm(len(lines))
        nb_train = int(len(lines) * (1 - valid_ratio))
        train_indices = indices[:nb_train]
        valid_indices = indices[nb_train:]

        for i in train_indices:
            train_split['name_text'].append(lines[i])
            train_split['name'].append(lines_torch[i])
            train_split['category_id'].append(category_id)
            train_split['category_text'].append(category)
            train_split['index'].append(i)

        for i in valid_indices:
            valid_split['name_text'].append(lines[i])
            valid_split['name'].append(lines_torch[i])
            valid_split['category_id'].append(category_id)
            valid_split['category_text'].append(category)
            valid_split['index'].append(i)

    valid_split['index'] = torch.tensor(valid_split['index'])
    train_split['index'] = torch.tensor(train_split['index'])

    sampler_train = SamplerRandom(batch_size=batch_size)
    sequence_train = SequenceArray(train_split, sampler=sampler_train).collate()

    sampler_valid = SamplerRandom(batch_size=batch_size)
    sequence_valid = SequenceArray(valid_split, sampler=sampler_valid).collate()

    return {
        'name_nationality': collections.OrderedDict([
            ('train', sequence_train),
            ('valid', sequence_valid),
        ])
    }
