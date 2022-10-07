from .callback import Callback
from ..train import utilities
from ..train import analysis_plots
import os
import collections
import logging
import numbers
from ..utils import safe_lookup

logger = logging.getLogger(__name__)


def extract_from_history(history, dataset_name, output_name, value_name):
    """
    Extract a specified value for all the epochs and given the dataset, split, output and value_name
    :param history: the history
    :param dataset_name: the dataset
    :param output_name: the output name
    :param value_name: the values to extract
    :return: a dictionary of list (split_name, values)
    """
    d = collections.defaultdict(list)
    for epoch_n, epoch in enumerate(history):
        dataset = epoch.get(dataset_name)
        if dataset is not None:
            for split_name, split in dataset.items():
                output = split.get(output_name)
                if output is not None:
                    value = output.get(value_name)
                    if value is not None and isinstance(value, numbers.Number):
                        d[split_name].append((epoch_n, value))

    return d


def merge_history_values(history_values_list):
    """
    Merge several history values (e.g., multiple runs)
    :param history_values_list: a list of history values
    :return: a dictionary of list of list of values (split name, list of history values)
    """
    d = collections.defaultdict(list)
    for history_values in history_values_list:
        for split_name, values in history_values.items():
            d[split_name].append(values)
    return d


def default_dicarded_metrics():
    return (),


class CallbackExportHistory(Callback):
    """
    Summarize the training history of a model (i.e., as a function of iteration)

    - One plot per dataset
    - splits are plotted together
    """
    def __init__(self, export_dirname='history', dicarded_metrics=default_dicarded_metrics()):
        self.export_dirname = export_dirname
        self.dicarded_metrics = dicarded_metrics

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('CallbackExportHistory.__call__ started')
        export_root = options.workflow_options.current_logging_directory
        sample_root_dir = os.path.join(export_root, self.export_dirname)
        utilities.create_or_recreate_folder(sample_root_dir)

        for dataset_name, _ in outputs.items():
            metrics_names = set()
            output_names = set()
            
            # find all possible metric names & output names
            for h in history:
                h_dataset = h.get(dataset_name)
                if h_dataset is not None:
                    for _, h_split in h_dataset.items():
                        for output_name, output in h_split.items():
                            output_names.add(output_name)
                            for metric_name, _ in output.items():
                                metrics_names.add(metric_name)

            # extract the values and graph them
            for output_name in output_names:
                for metric_name in metrics_names:
                    r = extract_from_history(history, dataset_name, output_name, metric_name)
                    if r is None:
                        continue
                    r = merge_history_values([r])
                    analysis_plots.plot_group_histories(
                        sample_root_dir,
                        r,
                        '{}/{}/{}'.format(dataset_name, output_name, metric_name),
                        xlabel='Epochs',
                        ylabel=metric_name)

        logger.info('CallbackExportHistory.__call__ done!')


