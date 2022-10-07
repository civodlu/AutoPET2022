from typing import List, Callable, Dict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from .store import RunResult, Metrics
from ..train import utilities
from ..train import analysis_plots
import math
import numbers
import logging


logger = logging.getLogger(__name__)


def is_discrete(values):
    """
    Test if a list of values is discrete or contiguous
    :param values: the list to test
    :return: True if discrete, False else
    """
    for value in values:
        if isinstance(value, np.ndarray) and len(value.shape) == 0:
            value = np.ndarray.item(value)
            
        if not isinstance(value, numbers.Number):
            return True
        frac, whole = math.modf(value)
        if frac != 0.0:
            return False
    return True


def median_by_category(categories, values):
    """
    Calculate the median for each categorical attribute
    :param categories: the categories
    :param values: the values
    :return: list of tuple (category, median value)
    """
    assert len(categories) == len(values)
    max_category = np.max(categories)
    min_category = np.min(categories)
    kvps = []
    for c in range(min_category, max_category + 1):
        c_indices = np.where(categories == c)
        if len(c_indices) == 0 or len(c_indices[0]) == 0:
            median = None
        else:
            median = np.median(values[c_indices])
        kvps.append((c, median))
    return kvps


def _plot_scatter(plot_name, x_values, x_name, y_values, y_name, discrete_random_jitter=0.2, x_ticks=None, y_ticks=None, median_max_num=20):
    """
    scatter plot with optional named ticks (x, y) and median display (x, y)
    :param plot_name:
    :param x_values:
    :param x_name:
    :param y_values:
    :param y_name:
    :param discrete_random_jitter:
    :param x_ticks:
    :param y_ticks:
    :param median_max_num:
    :return:
    """
    fig = plt.figure()
    subplot = 111
    ax = fig.add_subplot(subplot)
    x_medians_kvp = None
    y_medians_kvp = None

    x_values_orig = x_values
    y_values_orig = y_values

    x_discrete = is_discrete(x_values)
    y_discrete = is_discrete(y_values)
    
    # make sure we really have discrete values (e.g., we could have 1.0, 2.0...)
    if x_discrete:
        x_values = np.asarray(x_values, np.int)
    if y_discrete:
        y_values = np.asarray(y_values, np.int)

    if x_discrete and not y_discrete:
        x_medians_kvp = median_by_category(x_values, y_values)

    if y_discrete and not x_discrete:
        y_medians_kvp = median_by_category(y_values, x_values)

    if x_discrete:
        # add a jitter to help with the understanding of the value distribution. For discrete values
        # it doesn't matter if we add a small displacement
        x_values = x_values + (np.random.rand(len(x_values)) - 0.5) * 2.0 * discrete_random_jitter

    if y_discrete:
        y_values = y_values + (np.random.rand(len(y_values)) - 0.5) * 2.0 * discrete_random_jitter

    ax.scatter(x_values, y_values)
    if x_discrete and x_ticks is None:
        # default scale for discrete values so that we can plot accurately the median
        min_value = int(min(x_values_orig))
        max_value = int(max(x_values_orig))
        
        # minor = True, as me may have hundreds of labels and we would not be able to read
        # any.
        ax.set_xticks(list(range(min_value, max_value + 1)), minor=True)

    if y_discrete and y_ticks is None:
        # default scale for discrete values so that we can plot accurately the median
        min_value = int(min(y_values_orig))
        max_value = int(max(y_values_orig))
        ax.set_yticks(list(range(min_value, max_value + 1)), minor=True)

    if x_ticks is not None:
        # custom ticks for discrete categories
        ax.set_xticks(list(range(len(x_ticks))), minor=False)
        ticks_labels_kvp = sorted(list(x_ticks.items()), key=lambda x: x[0])
        ticks_labels = [name for _, name in ticks_labels_kvp]
        ax.set_xticklabels(ticks_labels, minor=False)

    if y_ticks is not None:
        # custom ticks for discrete categories
        ax.set_yticks(list(range(len(y_ticks))), minor=False)
        ticks_labels_kvp = sorted(list(y_ticks.items()), key=lambda x: x[0])
        ticks_labels = [name for _, name in ticks_labels_kvp]
        ax.set_yticklabels(ticks_labels, minor=False)

    if x_medians_kvp is not None and len(x_medians_kvp) < median_max_num:
        # only display when we don't have too many discrete values for legibility reasons
        # display the median values per discrete value
        for discrete_value, median in x_medians_kvp:
            ax.hlines(median, discrete_value - 0.3, discrete_value + 0.3)

    if y_medians_kvp is not None and len(y_medians_kvp) < median_max_num:
        # only display when we don't have too many discrete values for legibility reasons
        # display the median values per discrete value
        for discrete_value, median in y_medians_kvp:
            ax.vlines(median, discrete_value - 0.3, discrete_value + 0.3)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(plot_name)
    return fig


def _plot_importance(plot_name, x_names, y_values, y_name, y_errors=None, x_name='hyper-parameters'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.35

    r = np.arange(len(x_names))
    ax.bar(r, y_values, width, color='r', yerr=y_errors)
    ax.set_ylabel(y_name)
    ax.set_xlabel(x_name)
    ax.set_title(plot_name)
    ax.set_xticks(r + width / 2.0, minor=False)
    ax.set_xticklabels(x_names, rotation=40, ha='right')
    ax.set_xlim(-width)
    return fig


def _plot_param_covariance(plot_name, x_name, x_values, y_name, y_values, xy_values, discrete_random_jitter=0.2, x_ticks=None, y_ticks=None):
    if is_discrete(x_values):
        x_values = x_values + (np.random.rand(len(x_values)) - 0.5) * 2.0 * discrete_random_jitter

    if is_discrete(y_values):
        y_values = y_values + (np.random.rand(len(y_values)) - 0.5) * 2.0 * discrete_random_jitter

    plt.close('all')
    fig = plt.figure()
    cm = plt.cm.get_cmap('cool')

    ax = fig.add_subplot(111)
    im = ax.scatter(x_values, y_values, c=xy_values, cmap=cm)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(plot_name)
    plt.colorbar(im)
    return fig


def discretize(values):
    """
    Map string to a int and record the mapping
    :param values:
    :return: (values, mapping)
    """
    assert len(values.shape) == 2
    nb_features = values.shape[1]
    mapping = {}

    new_values = []
    for feature_id in range(nb_features):
        feature_values = values[:, feature_id]
        if isinstance(feature_values[0], (np.string_, str)):
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(values[:, 0])
            mapping[feature_id] = dict(enumerate(encoder.classes_))
            new_values.append(encoded_values)
        else:
            new_values.append(feature_values)
    return np.stack(new_values, axis=1), mapping


def analyse_hyperparameters(run_results: List[RunResult],
                            output_path: str,
                            loss_fn: Callable[[Metrics], float] = lambda metrics: metrics['loss'],
                            hparams_to_visualize: List[str] = None,
                            params_forest_n_estimators: int = 5000,
                            params_forest_max_features_ratio: float = 0.6,
                            top_k_covariance: int = 5,
                            create_graphs: bool = True,
                            verbose: bool = True,
                            dpi: int = 300) -> Dict[str, List]:
    """
    Importance hyper-parameter estimation using random forest regressors.

    From simulation, the ordering of hyper-parameters importance is correct, but the importance value itself may be
    over-estimated (for the best param) and underestimated (for the others).

    The scatter plot for each hyper parameter is useful to understand in what direction the
    hyper-parameter should be modified.

    The covariance plot can be used to understand the relation between most important hyper-parameter.

    WARNING:
        With correlated features, strong features can end up with low scores and the method can be biased towards
        variables with many categories. See for more details [1]_, [2]_.

    .. [1] http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
    .. [2] https://link.springer.com/article/10.1186%2F1471-2105-8-25

    Args:
        run_results: a list of runs
        output_path: where to export the graphs
        loss_fn: a function to extract a single value (loss) from a list of metrics
        hparams_to_visualize: a list of parameters (string) to visualize
        params_forest_n_estimators: number of trees used to estimate the loss from the hyperparameters
        params_forest_max_features_ratio: the maximum number of features to be used. Note we don't want to
           select all the features to limit the correlation importance decrease effect [1]_
        top_k_covariance: export the parameter covariance for the most important k hyper-parameters
        create_graphs: if True, export matplotlib visualizations
        verbose: if True, display additional information
        dpi: the resolution of the exported graph

    Returns:
        2 lists representing the hyper parameter name and importance

    """
    data = []
    for run_result in run_results:
        loss = loss_fn(run_result.metrics)
        if loss is None:
            # we don't want to analyze the run that could not
            # calculate the metric
            continue
        params = run_result.hyper_parameters

        def to_value(v):
            if isinstance(v, numbers.Number):
                return v
            return str(v)

        params_current = {}
        if hparams_to_visualize is None:
            for key, value in params.hparams.items():
                params_current[key] = to_value(value.current_value)
        else:
            for key in hparams_to_visualize:
                v = params.hparams.get(key)
                if v is not None:
                    params_current[key] = to_value(v)

        params_current['loss'] = loss
        data.append(params_current)

    f = pd.DataFrame(data)
    param_names = [name for name in f.columns if name != 'loss']

    # hyper parameter importance: use an extra tree to predict the loss value from the hyper parameter values
    # in order to calculate the importance of each feature
    Estimator = ExtraTreesRegressor
    params_forest = {
        'n_estimators': params_forest_n_estimators,
        'max_features': max(1, int(params_forest_max_features_ratio * len(param_names))),
    }
    forest = Estimator(**params_forest)

    values = f[param_names].values
    values, mapping = discretize(values)  # here special case for strings as values
    forest.fit(X=values, y=f['loss'].values)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    sorted_param_names = np.asarray(param_names)[indices]
    sorted_importances = importances[indices]

    if create_graphs:
        if verbose:
            print('output_path=%s' % output_path)
            
        utilities.create_or_recreate_folder(output_path)

        fig = _plot_importance(plot_name='hyper-parameter importance',
                               x_names=sorted_param_names,
                               y_values=sorted_importances,
                               y_name='importance',
                               y_errors=std)
        fig.savefig(os.path.join(output_path, 'hyper-parameter importance.png'), dpi=dpi)

        # now we know what parameter is important, but we need to understand in what direction this is important
        # (e.g., beneficial or detrimental values?)
        for h in indices:
            param_name = param_names[h]
            fig = _plot_scatter(plot_name='[%s] variations' % param_name,
                                y_values=f['loss'].values,
                                y_name='loss',
                                x_name=param_name,
                                x_values=values[:, h],
                                x_ticks=mapping.get(h))
            fig.savefig(os.path.join(output_path, param_name + '.png'), dpi=dpi)

        # finally, we want to look at the hyper parameter covariances
        best_param_names = sorted_param_names[:top_k_covariance]
        for y in range(0, len(param_names)):
            for x in range(y + 1, len(param_names)):
                feature_1 = param_names[y]
                feature_2 = param_names[x]
                if feature_1 in best_param_names and feature_2 in best_param_names:
                    fig = _plot_param_covariance(plot_name='[%s, %s] variations' % (feature_1, feature_2),
                                                 y_values=values[:, y],
                                                 y_name=feature_1,
                                                 x_values=values[:, x],
                                                 x_name=feature_2,
                                                 xy_values=f['loss'])
                    fig.savefig(os.path.join(output_path, 'covariance_' + feature_1 + '_' + feature_2 + '.png'), dpi=dpi)

    return {
        'sorted_param_names': sorted_param_names,
        'sorted_importances': sorted_importances
    }
