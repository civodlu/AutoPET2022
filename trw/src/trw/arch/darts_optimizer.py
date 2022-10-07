import collections
import torch
from . import darts_cell
import functools


def _get_parameters(model, is_darts_weight_dataset_name):
    """
    Capture the parameters relative to DARTS & dataset
    """
    parameters = []

    if is_darts_weight_dataset_name:
        for parameter in model.parameters():
            if isinstance(parameter, darts_cell.SpecialParameter):
                parameters.append(parameter)
    else:
        for parameter in model.parameters():
            if not isinstance(parameter, darts_cell.SpecialParameter):
                parameters.append(parameter)

    return parameters


def create_darts_optimizers_fn(datasets, model, optimizer_fn, darts_weight_dataset_name, scheduler_fn=None):
    """
    Create an optimizer and scheduler for DARTS architecture search.

    In particular, parameters that are derived from :class:`trw.arch.SpecialParameter` will be handled differently:

        - for each dataset that is not equal to `darts_weight_dataset_name`, optimize all the parameters not
            derived from :class:`trw.arch.SpecialParameter`

        - on the dataset `darts_weight_dataset_name`, ONLY the parameters derived from :class:`trw.arch.SpecialParameter`
            will be optimized


    Note:
        if model is an instance of`ModuleDict`, then the optimizer will only consider the parameters
        `model[dataset_name].parameters()` else `model.parameters()`

    Args:
        datasets: a dictionary of dataset
        model: the model. Should be a `Module` or a `ModuleDict`
        optimizer_fn: the functor to instantiate the optimizer
        scheduler_fn: the functor to instantiate the scheduler. May be None, in that case
            there will be no scheduler
        darts_weight_dataset_name: this specifies the dataset to be used to train the DARTS cell
            weights. Only the parameters of the model derived from :class:`trw.arch.SpecialParameter`
            will be optimized on the dataset `darts_weight_dataset_name`

    Returns:
        a dict of optimizers, one per dataset
    """

    schedulers = None
    if scheduler_fn is not None:
        schedulers = collections.OrderedDict()
    optimizers = collections.OrderedDict()
    for dataset_name in datasets.keys():
        is_darts_weight_dataset_name = dataset_name == darts_weight_dataset_name
        if isinstance(model, torch.nn.ModuleDict):
            # this is a collection of model. Assumed we have a different model
            # per dataset to be optimized
            sub_model = model[dataset_name]
            optimizer = optimizer_fn(_get_parameters(sub_model, is_darts_weight_dataset_name=is_darts_weight_dataset_name))
        else:
            optimizer = optimizer_fn(_get_parameters(model, is_darts_weight_dataset_name=is_darts_weight_dataset_name))

        optimizers[dataset_name] = optimizer

        if schedulers is not None:
            scheduler = scheduler_fn(optimizer)
            schedulers[dataset_name] = scheduler

    return optimizers, schedulers, None


def create_darts_adam_optimizers_fn(datasets, model, darts_weight_dataset_name, learning_rate, scheduler_fn=None):
    """
    Create an ADAM optimizer and scheduler for DARTS architecture search.

    Args:
        datasets: a dictionary of dataset
        model: a model to optimize
        learning_rate: the initial learning rate
        scheduler_fn: a scheduler, or `None`
        darts_weight_dataset_name: this specifies the dataset to be used to train the DARTS cell
            weights. Only the parameters of the model derived from :class:`trw.arch.SpecialParameter`
            will be optimized on the dataset `darts_weight_dataset_name`

    Returns:
        An optimizer
    """
    optimizer_fn = functools.partial(torch.optim.Adam, lr=learning_rate)
    return create_darts_optimizers_fn(
        datasets=datasets,
        model=model,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        darts_weight_dataset_name=darts_weight_dataset_name)

