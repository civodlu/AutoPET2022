from collections import defaultdict
import json
import logging
import os
import time
from typing import Iterator
import trw
from trw.basic_typing import Batch
from trw.callbacks.callback import Callback
import numpy as np
from trw.utils import safe_filename
import matplotlib.pyplot as plt
import pandas as pd
from .callback_experiment_tracking import get_git_revision
from corelib import compare_volumes_mips
import h5py


logger = logging.getLogger(__name__)


def find_root_sequence(seq: trw.train.Sequence) -> trw.train.Sequence:
    if hasattr(seq, 'source_split') and seq.source_split is not None:
        return find_root_sequence(seq.source_split)
    return seq

def get_data_inmemory_fn(split: trw.train.Sequence) -> Iterator[Batch]:
    root_sequence = find_root_sequence(split)
    assert isinstance(root_sequence, trw.train.sequence_array.SequenceArray)
    return root_sequence

def write_hdf5(path, o: np.ndarray, origin, spacing, chunk_shape=(32, 32, 32)):
    with h5py.File(path, 'w') as f:
        f.create_dataset('v', data=o, chunks=chunk_shape, compression='lzf')
        f.create_dataset('origin', data=origin, compression='lzf')
        f.create_dataset('spacing', data=spacing, compression='lzf')

class CallbackInference(Callback):
    """
    Run the wholebody inference
    """
    def __init__(
            self, 
            inference_fn,
            get_data_fn=get_data_inmemory_fn, 
            export_3d_volumes=False, 
            output_dir_name='wholebody_inference',
            #cmap=plt.get_cmap('plasma'),
            cmap=plt.get_cmap('binary'),
            show_error=False,
            flip_mips=True,
            metrics_fn=None,
            result_path=None,
            max_value_mip=None,
            max_value_name=None,
            mip_output_scaling=1.0,
            skip_train=True,
            export_inference_prob=False,
            ) -> None:
        """

        Args:
            max_value_mip: if `max_value_mip` is None, None or a default max value for the generated MIP
            max_value_name: find in the batch `max_value_name` to be used as max value
        """
        super().__init__()
        self.get_data_fn = get_data_fn
        self.inference_fn = inference_fn
        self.export_3d_volumes = export_3d_volumes
        self.export_inference_prob = export_inference_prob
        self.output_dir_name = output_dir_name
        self.cmap = cmap
        self.flip_mips = flip_mips
        self.show_error = show_error
        self.metrics_fn = metrics_fn
        self.result_path = result_path
        self.max_value_mip = max_value_mip
        self.max_value_name = max_value_name
        self.mip_output_scaling = mip_output_scaling
        self.skip_train = skip_train

    def __call__(self, options, history, model, losses, outputs, datasets, datasets_infos, callbacks_per_batch, **kwargs):
        logger.info('Started CallbackInference!')

        metrics_datasets = defaultdict(list)
        output_root = os.path.join(options.workflow_options.current_logging_directory, self.output_dir_name)
        os.makedirs(output_root, exist_ok=True)
        for dataset_name, dataset in datasets.items():
            for split_name, split in dataset.items():
                if self.skip_train and split_name == 'train':
                    continue

                root_sequence = self.get_data_fn(split)
                for batch_n, batch in enumerate(root_sequence):
                    uid = batch['case_name']
                    logger.info(f'Processing case={batch_n}, uid={uid}')
                    inference_time_start = time.perf_counter()
                    inference_output = self.inference_fn(batch, model)
                    inference_time_end = time.perf_counter()

                    safe_uid = safe_filename(uid)
                    basename = os.path.join(output_root, f'{dataset_name}_{split_name}_{safe_uid}')

                    if self.export_3d_volumes:
                        write_hdf5(basename + 'output_found.hdf5', inference_output.output_found.numpy(), origin=batch['current_origin'], spacing=batch['current_spacing'])
                        write_hdf5(basename + 'output_truth.hdf5', inference_output.output_truth.numpy(), origin=batch['current_origin'], spacing=batch['current_spacing'])
                        write_hdf5(basename + 'output_input.hdf5', inference_output.output_input.numpy(), origin=batch['current_origin'], spacing=batch['current_spacing'])

                    if self.export_inference_prob:
                        # we only need the prob of lesion
                        assert len(inference_output.output_raw.shape) == 4, 'must be NDHW format!'
                        if inference_output.output_raw.shape[0] == 2:
                            write_hdf5(basename + 'output_raw.hdf5', inference_output.output_raw[1].numpy(), origin=batch['current_origin'], spacing=batch['current_spacing'])
                        else:
                            write_hdf5(basename + 'output_raw.hdf5', inference_output.output_raw.numpy(), origin=batch['current_origin'], spacing=batch['current_spacing'], chunk_shape=(16, 32, 32, 32))
                            # we don't want any MIP or stats here, this is just
                            # to collect internal features!
                            continue
                    #
                    # Export MIPs
                    #
                    volumes = [[
                        inference_output.output_input.numpy(),
                        inference_output.output_found.numpy() * self.mip_output_scaling,
                        inference_output.output_truth.numpy() * self.mip_output_scaling
                    ]]

                    if self.show_error:
                        error = (inference_output.output_found - inference_output.output_truth).abs()
                        volumes[0].append(error)

                    case_names = [safe_uid]
                    category_names = ['SUV', 'SEG found', 'SEG Truth']
                    if self.show_error:
                        category_names.append('error')

                    #
                    # collect metrics
                    #
                    metrics_datasets['dataset_split'].append(f'{dataset_name}/{split_name}')
                    metrics_datasets['case_uid'].append(uid)
                    metrics_datasets['recon_time'].append(inference_time_end - inference_time_start)

                    # dice
                    caption = ''
                    if self.metrics_fn is not None:
                        additional_metrics = self.metrics_fn(
                            found=inference_output.output_found,
                            truth=inference_output.output_truth)
                        for name, value in additional_metrics.items():
                            if value is not None:
                                caption = caption + f' {name}:{value:.2f}'
                            else:
                                caption = caption + f' {name}:None'
                            metrics_datasets[name].append(value)

                    # configure the max value to be displayed
                    # for the MIP. This is important, if too high,
                    # we won't see anything...
                    if self.max_value_name is not None and self.max_value_name in batch:
                        max_value_mip = batch[self.max_value_name]
                    elif self.max_value_mip is not None:
                        max_value_mip = self.max_value_mip
                    else:
                        max_value_mip = None

                    fig = compare_volumes_mips(
                        volumes, 
                        case_names, 
                        category_names, 
                        self.cmap, 
                        flip=self.flip_mips, 
                        with_xy=False,
                        max_value=max_value_mip,
                        title=caption,
                        overlay_with=[None, 0, 0]
                    )
                    fig.savefig(basename + f'-e{len(history)}.png')
                    plt.close(fig)

                    log_str = f'case={uid}, metrics={additional_metrics}, time={inference_time_end - inference_time_start}'
                    print(log_str)
                    logger.info(log_str)


        # collect metrics
        df = pd.DataFrame(metrics_datasets)
        if df.size == 0:
            logger.info('No statistics collected!')
            return 

        aggregate_by_dataset = df.groupby(['dataset_split']).agg('mean').to_dict()
        experiment_name = os.path.basename(options.workflow_options.current_logging_directory)

        # summary in the log
        log_str = f'{aggregate_by_dataset}'
        print(log_str)
        logger.info(log_str)

        if len(history) > 0:
            # record stats in the history
            history_split = trw.utils.safe_lookup(history[-1], dataset_name, split_name)
            if history_split is not None:
                history_step = {}
                for name, d in aggregate_by_dataset.items():
                    value = d.get(f'{dataset_name}/{split_name}')
                    if value is not None:
                        history_step[name] = value
                history_split['autopet'] = history_step

        def save_metrics(metrics_all, metrics_aggregated, path):
            configuration = options.runtime.configuration

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'w') as f:
                json.dump({
                    'aggregated': metrics_aggregated, 
                    'epoch': len(history),
                    'experiment_name': experiment_name,
                    'git_hash': get_git_revision(),
                    'derived_from': configuration.tracking['derived_from'],
                    'datasets_version': configuration.datasets['datasets_version'],
                    'splits_name': configuration.datasets['splits_name'],
                    # make it last, the aggregated header needs to be on
                    # top to make it easier to review without scrolling too much
                    'raw_data': metrics_all,
                    }, f, indent=3)


        save_metrics(metrics_datasets, aggregate_by_dataset, os.path.join(options.workflow_options.current_logging_directory, 'metrics.json'))
        if self.result_path is not None:
            here = os.path.abspath(os.path.dirname(__file__))
            save_metrics(metrics_datasets, aggregate_by_dataset, os.path.join(here, self.result_path, f'{experiment_name}.json'))
        logger.info('finished CallbackInference!')