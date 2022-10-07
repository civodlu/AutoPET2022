"""
Here we implement useful default hyper-parameter creators that are registered
in the :class:`trw.hparams.HyperParameterRepository`
"""
import logging
from typing import Sequence, Tuple, Optional
import torch
from torch import nn

from typing_extensions import Literal


from .params import create_discrete_value, create_boolean, create_continuous_power
from ..train.optimizers import create_sgd_optimizers_fn, create_adam_optimizers_fn
from ..basic_typing import Datasets, ModuleCreator
from ..layers.layer_config import NormType, PoolType


logger = logging.getLogger(__name__)


def create_optimizers_fn(
        datasets: Datasets,
        model: nn.Module,
        optimizers: Sequence[Literal['adam', 'sgd']] = ('adam', 'sgd'),  # type: ignore  # TODO revisit this one
        lr_range: Tuple[float, float, float] = (1e-3, -5, -1),
        momentum: Sequence[float] = (0.5, 0.9, 0.99),
        beta_1: Sequence[float] = (0.9,),
        beta_2: Sequence[float] = (0.999, 0.99),
        eps: Sequence[float] = (1e-8,),
        weight_decay: Optional[Sequence[float]] = (0.0, 1e-4, 1e-5, 1e-6, 1e-8),
        name_prefix='trw.') -> torch.optim.Optimizer:
    """
    Create hyper-parameters for a wide range of optimizer search.

    Hyper-parameters will be named using 2 groups of hyper-parameters:
    - `trw.optimizers.*`: most important hyper-parameters to search
    - `trw.optimizers_fine.*`: hyper-parameters that we might want to search but in most cases
        would not significantly influence the results. These hyper-parameters maybe discarded during
        the hyper-parameter optimization

    Args:
        datasets: the datasets
        model: the model to be optimized
        optimizers: the optimizers to search
        lr_range: the learning rate range (min, max)
        momentum: the momentum values to test
        beta_1: the beta_1 values to test
        beta_2: the beta_2 values to test
        eps: the epsilon values to test
        weight_decay: the weight decay values to test
        name_prefix: prefix appended to the hyper-parameter name

    Returns:
        A dict of optimizer per dataset
    """
    optimizer_name = create_discrete_value(name_prefix + 'optimizers', default_value=optimizers[0], values=list(optimizers))

    if optimizer_name == 'sgd':
        learning_rate = create_continuous_power(name_prefix + 'optimizers.sgd.lr', lr_range[0], lr_range[1], lr_range[2])
        momentum = create_discrete_value(name_prefix + 'optimizers.sgd.momentum', momentum[0], list(momentum))
        nesterov = create_boolean(name_prefix + 'optimizers_fine.sgd.nesterov', True)
        if weight_decay is not None and len(weight_decay) > 0:
            w = create_discrete_value(
                name_prefix + 'optimizers_fine.sgd.weight_decay',
                weight_decay[0],
                list(weight_decay))
        else:
            w = 0
        return create_sgd_optimizers_fn(
            datasets=datasets,
            model=model,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=w,
            nesterov=nesterov
        )
    elif optimizer_name == 'adam':
        learning_rate = create_continuous_power(name_prefix + 'optimizers.adam.lr', lr_range[0], lr_range[1], lr_range[2])
        b1 = create_discrete_value(name_prefix + 'optimizers_fine.adam.beta_1', beta_1[0], list(beta_1))
        b2 = create_discrete_value(name_prefix + 'optimizers_fine.adam.beta_2', beta_2[0], list(beta_2))
        e = create_discrete_value(name_prefix + 'optimizers_fine.adam.eps', eps[0], list(eps))
        if weight_decay is not None and len(weight_decay) > 0:
            w = create_discrete_value(
                name_prefix + 'optimizers_fine.adam.weight_decay',
                weight_decay[0],
                list(weight_decay))
        else:
            w = 0
        return create_adam_optimizers_fn(
            datasets=datasets,
            model=model,
            learning_rate=learning_rate,
            betas=(b1, b2),
            weight_decay=w,
            eps=e
        )
    else:
        raise ValueError(f'unhandled value={optimizer_name}')


def create_activation(
        name: str,
        default_value: nn.Module,
        functions: Sequence[ModuleCreator] = (
                nn.ReLU,
                nn.ReLU6,
                nn.LeakyReLU,
                nn.ELU,
                nn.PReLU,
                nn.RReLU,
                nn.SELU,
                nn.CELU,
                #nn.GELU,  # TODO torch >= 1.4
                nn.Softplus
        )) -> nn.Module:
    """
    Create activation functions

    Args:
        name: the name of the hyper-parameter
        functions: the activation functions
        default_value: the default value at creation

    Returns:
        a functor to create the activation function
    """
    assert default_value in functions, f'missing value={default_value} in possible values!'
    return create_discrete_value(name, default_value=default_value, values=list(functions))


def create_norm_type(
        name: str,
        default_value: Optional[NormType],
        norms: Sequence[Optional[NormType]] = (
                NormType.BatchNorm,
                NormType.InstanceNorm,
                None,)) -> NormType:
    """
    Create a normalization layer type hyper-parameter

    Args:
        name: the name of the hyper-parameter
        norms: a sequence of :class:`NormType`
        default_value: the default value at creation

    Returns:
        a normalization layer type
    """
    assert default_value in norms
    return create_discrete_value(name, default_value=default_value, values=list(norms))


def create_pool_type(
    name: str,
    default_value: PoolType,
    pools: Sequence[PoolType] = (
            PoolType.MaxPool,
            PoolType.AvgPool,
            PoolType.FractionalMaxPool
        )) -> PoolType:
    """
    Create a pooling type hyper-parameter
    Args:
        name: the name of the hyper-parameter
        pools: the available pooling types
        default_value: the default value at creation

    Returns:
        a pooling type
    """
    assert default_value in pools
    return create_discrete_value(name, default_value=default_value, values=list(pools))
