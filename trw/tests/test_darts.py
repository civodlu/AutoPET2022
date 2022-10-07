from unittest import TestCase
import trw
import numpy as np
import collections
import math
import torch
import torch.nn as nn
import functools


class SinModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class ExpModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class ReluModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)


def create_darts_dataset_1d(fn=math.sin, range=np.arange(0, 2.0, 0.05), training_params_ratio=0.8, batch_size=1):
    """
    Create a dataset suitable for synthetic DARTS tests
    """
    values_x = []
    values_fx = []

    for x in range:
        fx = fn(x)
        values_x.append(x)
        values_fx.append(fx)

    values_x = np.asarray(values_x, dtype=np.float32)
    values_fx = np.asarray(values_fx, dtype=np.float32)

    indices = np.arange(len(values_x))
    np.random.shuffle(indices)

    nb_training_params = int(training_params_ratio * len(indices))

    sampler = trw.train.SamplerSequential(batch_size=batch_size)
    training_params = trw.train.SequenceArray({
        'x': values_x[indices[:nb_training_params]],
        'fx': values_fx[indices[:nb_training_params]],
    }, sampler=sampler).collate()

    sampler = trw.train.SamplerSequential(batch_size=batch_size)
    training_weights = trw.train.SequenceArray({
        'x': values_x[indices[nb_training_params:]],
        'fx': values_fx[indices[nb_training_params:]],
    }, sampler=sampler).collate()

    datasets = collections.OrderedDict()
    datasets['train_params'] = {'train': training_params}
    datasets['training_weights'] = {'train': training_weights}
    return datasets


def darts_fn_1(x, bias=1.0):
    return math.sin(x) + bias


def primitive_linear(_1, _2, _3):
    return nn.Linear(1, 1)


def primitive_sin(_1, _2, _3):
    return SinModule()


def primitive_exp(_1, _2, _3):
    return ExpModule()


def primitive_relu(_1, _2, _3):
    return ReluModule()

def get_primitives():
    return collections.OrderedDict([
        ('linear', primitive_linear),
        ('sin', primitive_sin),
        # ('exp', primitive_exp),
        # ('relu', primitive_relu),
    ])


class ModelDarts1D_fn_1(nn.Module):
    """
    we try to model a function `sin(x)` + bias, where the model
    has to learn the bias and DARTS has to learn the correct function (`sin`)
    """
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

        primitives = get_primitives()

        cell_merge_output_fn = functools.partial(trw.arch.default_cell_output, nb_outputs_to_use=1)
        self.cell = trw.arch.Cell(
            primitives=primitives,
            cpp=1, cp=1, c=1,
            is_reduction=False,
            is_reduction_prev=False,
            internal_nodes=1,
            with_preprocessing=False,
            cell_merge_output_fn=cell_merge_output_fn)

    def forward(self, batch):
        x = batch['x']
        x = x.unsqueeze(1)
        x = self.cell([x, x])
        x += self.bias
        return {
            'output_fn': trw.train.OutputRegression(x, batch['fx'])
        }


class TestDarts(TestCase):
    def test_parameter_known_solution(self):
        """
        Create a synthetic problem with a single best value for the
        cell weights and the NN weights. Make sure we find the expected solution
        """
        options = trw.train.Options(num_epochs=50)
        trainer = trw.train.TrainerV2()
        model = ModelDarts1D_fn_1()
        r = trainer.fit(
            options=options,
            datasets=create_darts_dataset_1d(),
            model=model,
            optimizers_fn=lambda datasets, model: trw.arch.create_darts_adam_optimizers_fn(datasets, model, darts_weight_dataset_name='training_weights', learning_rate=0.01)
        )

        # this represents the architecture of a cell
        # Now, create a cell that implements this architecture
        genotype = model.cell.get_genotype()

        # TODO  when we handle variable cell inputs, make sure we find the expected optimized parameters!

        cell_merge_output_fn = functools.partial(trw.arch.default_cell_output, nb_outputs_to_use=1)
        cell = trw.arch.Cell(
            primitives=get_primitives(),
            cpp=1, cp=1, c=1,
            is_reduction=False,
            is_reduction_prev=False,
            internal_nodes=1,
            with_preprocessing=False,
            cell_merge_output_fn=cell_merge_output_fn,
            genotype=genotype)

        # make sure the implementation has actually the same genotyper as the original!
        genotype_2 = cell.get_genotype()
        assert genotype_2 == genotype, 'the reconstruction must have the same genotype as the original!'

