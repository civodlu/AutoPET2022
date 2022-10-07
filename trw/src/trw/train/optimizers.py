import functools
import torch
import collections

from ..utils import torch_requires


def create_scheduler_step_lr(optimizer, step_size=30, gamma=0.1):
    """
    Create a learning rate scheduler. Every `step_size`, the learning late will be multiplied by `gamma`

    Args:
        optimizer: the optimizer
        step_size: every number of epochs composing one step. Each step the learning rate will be decreased
        gamma: apply this factor to the learning rate every time it is adjusted

    Returns:
        a learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def create_optimizers_fn(datasets, model, optimizer_fn, scheduler_fn=None, per_step_scheduler_fn=None):
    """
    Create an optimizer and scheduler

    Note:
        if model is an instance of`ModuleDict`, then the optimizer will only consider the parameters
        `model[dataset_name].parameters()` else `model.parameters()`

    Args:
        datasets: a dictionary of dataset
        model: the model. Should be a `Module` or a `ModuleDict`
        optimizer_fn: the functor to instantiate the optimizer
        scheduler_fn: the functor to instantiate the scheduler to be run by epoch. May
            be None, in that case there will be no schedule
        per_step_scheduler_fn: the functor to instantiate scheduler to be run per-step (batch)
    """

    per_step_schedulers = None
    schedulers = None
    if scheduler_fn is not None:
        schedulers = collections.OrderedDict()
    if per_step_scheduler_fn is not None:
        per_step_schedulers = collections.OrderedDict()

    optimizers = collections.OrderedDict()
    for dataset_name in datasets.keys():
        if isinstance(model, torch.nn.ModuleDict):
            # this is a collection of model. Assumed we have a different model
            # per dataset to be optimized
            sub_model = model[dataset_name]
            optimizer = optimizer_fn(sub_model.parameters())
        else:
            optimizer = optimizer_fn(model.parameters())

        optimizers[dataset_name] = optimizer

        if schedulers is not None and optimizer is not None:
            scheduler = scheduler_fn(optimizer)
            schedulers[dataset_name] = scheduler

        if per_step_schedulers is not None and optimizer is not None:
            per_step_scheduler = per_step_scheduler_fn(optimizer)
            per_step_schedulers[dataset_name] = per_step_scheduler

    return optimizers, schedulers, per_step_schedulers


def create_adam_optimizers_fn(
        datasets,
        model,
        learning_rate,
        weight_decay=0,
        betas=(0.9, 0.999),
        eps=1e-8,
        scheduler_fn=None,
        per_step_scheduler_fn=None):
    """
    Create an ADAM optimizer for each of the dataset with optional scheduler

    Args:
        datasets: a dictionary of datasets
        model: a model to optimize
        learning_rate: the initial learning rate
        weight_decay: the weight decay
        scheduler_fn: a scheduler, or `None`
        betas: coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: term to add to denominator to avoid division by zero
        per_step_scheduler_fn: the functor to instantiate scheduler to be run per-step (batch)

    Returns:
        An optimizer
    """
    optimizer_fn = functools.partial(torch.optim.Adam, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)
    return create_optimizers_fn(
        datasets,
        model,
        optimizer_fn=optimizer_fn,
        scheduler_fn=scheduler_fn,
        per_step_scheduler_fn=per_step_scheduler_fn)


def create_adam_optimizers_scheduler_step_lr_fn(datasets, model, learning_rate, step_size, gamma, weight_decay=0, betas=(0.9, 0.999)):
    """
    Create an ADAM optimizer for each of the dataset with optional scheduler

    Args:
        datasets: a dictionary of dataset
        model: a model to optimize
        learning_rate: the initial learning rate
        step_size: the number of epoch composing a step. Each step the learning rate will be multiplied by `gamma`
        gamma: the factor to apply to the learning rate every step
        weight_decay : the weight decay
        betas: coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))

    Returns:
        An optimizer with a step scheduler
    """
    scheduler_fn = functools.partial(create_scheduler_step_lr, step_size=step_size, gamma=gamma)
    return create_adam_optimizers_fn(
        datasets,
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        scheduler_fn=scheduler_fn
    )


def create_sgd_optimizers_fn(datasets, model, learning_rate, momentum=0.9, weight_decay=0, nesterov=False, scheduler_fn=None, per_step_scheduler_fn=None):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with optional scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            learning_rate: the initial learning rate
            scheduler_fn: a scheduler, or `None`
            momentum: the momentum of the SGD
            weight_decay: the weight decay
            nesterov: enables Nesterov momentum
            per_step_scheduler_fn: the functor to instantiate scheduler to be run per-step (batch)

        Returns:
            An optimizer
        """
    optimizer_fn = functools.partial(
        torch.optim.SGD,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov)
    return create_optimizers_fn(datasets, model,
                                optimizer_fn=optimizer_fn,
                                scheduler_fn=scheduler_fn,
                                per_step_scheduler_fn=per_step_scheduler_fn)


def create_sgd_optimizers_scheduler_step_lr_fn(
        datasets,
        model,
        learning_rate,
        step_size,
        gamma,
        weight_decay=0,
        momentum=0.9,
        nesterov=False):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with step learning rate scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            learning_rate: the initial learning rate
            step_size: the number of epoch composing a step. Each step the learning rate will be multiplied by `gamma`
            gamma: the factor to apply to the learning rate every step
            weight_decay: the weight decay
            nesterov: enables Nesterov momentum
            momentum: the momentum of the SGD

        Returns:
            An optimizer with a step scheduler
        """
    scheduler_fn = functools.partial(create_scheduler_step_lr, step_size=step_size, gamma=gamma)
    return create_sgd_optimizers_fn(
        datasets,
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_fn=scheduler_fn,
        momentum=momentum,
        nesterov=nesterov)


@torch_requires(min_version='1.3')
def create_sgd_optimizers_scheduler_one_cycle_lr_fn(
        datasets,
        model,
        max_learning_rate,
        epochs,
        steps_per_epoch,
        additional_scheduler_kwargs=None,
        weight_decay=0,
        learning_rate_start_div_factor=25,
        learning_rate_end_div_factor=10000,
        percentage_cycle_increase=0.3,
        nesterov=False):
    """
        Create a Stochastic gradient descent optimizer for each of the dataset with step learning rate scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            max_learning_rate: the maximum learning rate
            epochs: The number of epochs to train for
            steps_per_epoch: The number of steps per epoch. If 0 or `None`, the schedule will be based on mumber of
                epochs only
            learning_rate_start_div_factor: defines the initial learning rate for the first step as
                initial_learning = max_learning_rate / learning_rate_start_div_factor
            learning_rate_end_div_factor: defines the end learning rate for the last step as
                final_learning_rate = max_learning_rate / learning_rate_start_div_factor / learning_rate_end_div_factor
            percentage_cycle_increase: The percentage of the cycle (in number of steps) spent
                increasing the learning rate
            additional_scheduler_kwargs: additional arguments provided to the scheduler
            weight_decay: the weight decay
            nesterov: enables Nesterov momentum
            momentum: the momentum of the SGD

        Returns:
            An optimizer with a step scheduler
        """
    scheduler_kwargs = {
        'div_factor': learning_rate_start_div_factor,
        'final_div_factor': learning_rate_end_div_factor,
        'pct_start': percentage_cycle_increase
    }
    if additional_scheduler_kwargs is None:
        additional_scheduler_kwargs = {}
    if scheduler_kwargs is not None:
        scheduler_kwargs = {**scheduler_kwargs, **additional_scheduler_kwargs}

    steps_per_epoch_effective = steps_per_epoch
    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch_effective = 1
        epochs += 1

    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_learning_rate,
            steps_per_epoch=steps_per_epoch_effective,
            epochs=epochs,
            **scheduler_kwargs
        )

    if steps_per_epoch_effective <= 1:
        kwargs = {'scheduler_fn': scheduler_fn}
    else:
        kwargs = {'per_step_scheduler_fn': scheduler_fn}

    return create_sgd_optimizers_fn(
        datasets,
        model,
        learning_rate=1.0,  # the scheduler will entirely manage the learning rate
        weight_decay=weight_decay,
        momentum=1.0,  # the scheduler will entirely manage the momentum
        nesterov=nesterov,
        **kwargs
    )


@torch_requires(min_version='1.3')
def create_adam_optimizers_scheduler_one_cycle_lr_fn(
        datasets,
        model,
        max_learning_rate,
        epochs,
        steps_per_epoch,
        additional_scheduler_kwargs=None,
        weight_decay=0,
        betas=(0.9, 0.999),
        eps=1e-8,
        learning_rate_start_div_factor=25,
        learning_rate_end_div_factor=10000,
        percentage_cycle_increase=0.3):
    """
        Create a ADAM optimizer for each of the dataset with step learning rate scheduler

        Args:
            datasets: a dictionary of dataset
            model: a model to optimize
            max_learning_rate: the maximum learning rate
            epochs: The number of epochs to train for
            steps_per_epoch: The number of steps per epoch. If 0 or `None`, the schedule will be based on mumber of
                epochs only
            learning_rate_start_div_factor: defines the initial learning rate for the first step as
                initial_learning = learning_rate_start_multiplier * max_learning_rate
            learning_rate_end_div_factor: defines the end learning rate for the last step as
                final_learning_rate = max_learning_rate / learning_rate_start_div_factor / learning_rate_end_div_factor
            percentage_cycle_increase: The percentage of the cycle (in number of steps) spent increasing
                the learning rate
            additional_scheduler_kwargs: additional arguments provided to the scheduler
            weight_decay: the weight decay
            betas: `betas` of the ADAM optimizer
            eps: `eps` of the ADAM optimizer

        Returns:
            An optimizer with a step scheduler
        """
    scheduler_kwargs = {
        'div_factor': learning_rate_start_div_factor,
        'final_div_factor': learning_rate_end_div_factor,
        'pct_start': percentage_cycle_increase
    }
    if additional_scheduler_kwargs is None:
        additional_scheduler_kwargs = {}
    if scheduler_kwargs is not None:
        scheduler_kwargs = {**scheduler_kwargs, **additional_scheduler_kwargs}

    steps_per_epoch_effective = steps_per_epoch
    if steps_per_epoch is None or steps_per_epoch == 0:
        steps_per_epoch_effective = 1
        epochs += 1

    scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_learning_rate,
            steps_per_epoch=steps_per_epoch_effective,
            epochs=epochs,
            **scheduler_kwargs
        )

    if steps_per_epoch_effective <= 1:
        kwargs = {'scheduler_fn': scheduler_fn}
    else:
        kwargs = {'per_step_scheduler_fn': scheduler_fn}

    return create_adam_optimizers_fn(
        datasets,
        model,
        learning_rate=1.0,  # the scheduler will entirely manage the learning rate
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        **kwargs)
