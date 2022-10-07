
import os
from torch import nn
from trw.basic_typing import Batch
from trw.train import Output
import time
from typing import Callable, Dict, Optional, Tuple
from corelib import resample_sitk_image, make_sitk_image, get_sitk_image_attributes, make_sitk_image_attributes
import numpy as np
from SimpleITK import GetArrayFromImage
import torch
import trw
import lz4
import pickle
from glob import glob
import copy



def get_case_files(root, case_name):
    matches = glob(os.path.join(root, f'*{case_name}output_found.npy'))
    assert len(matches) == 1, f'got={matches}'

    return {
        'cascade.inference.output_found': torch.from_numpy(np.load(matches[0]))
    }

class TransformCascadedPredictionPrecomputed:
    """
    The cascading is already on the filesystem. Just reload and resample to current spacing
    """
    def __init__(
            self,
            precomputed_root: str,
            case_name_to_files: Callable[[str, str], Dict] = get_case_files,
            precomputed_spacing_xyz: Tuple[float, float, float] = (6.0, 6.0, 6.0),
            ) -> None:
        self.precomputed_root = precomputed_root
        self.case_name_to_files = case_name_to_files
        self.precomputed_spacing_xyz = precomputed_spacing_xyz

    def __call__(self, batch: Batch) -> Batch:
        one_image_name = [n for n, v in batch.items() if isinstance(v, torch.Tensor) and len(v.shape) == 3]
        assert len(one_image_name) >= 1, 'no image found!'
        target_shape_xyz = batch[one_image_name[0]].shape[::-1]
        target_spacing_xyz = batch['original_spacing']

        case_name = batch['case_name']
        time_loading_start = time.perf_counter()
        features = self.case_name_to_files(self.precomputed_root, case_name)
        time_loading_end = time.perf_counter()

        time_resampling_start = time.perf_counter()
        for name, image in features.items():
            assert name not in batch, f'feature={name} already exists in the batch!'

            # resample the data tp original spacing
            image_sitk = make_sitk_image(
                image.numpy(), 
                origin_xyz=np.zeros([3]), 
                spacing_xyz=self.precomputed_spacing_xyz
            )
            fill_value = image.min()

            # remember SITK operates in XYZ not ZYX
            target_attributes = get_sitk_image_attributes(image_sitk)
            target_attributes['spacing'] = target_spacing_xyz
            target_attributes['shape'] = target_shape_xyz

            image_resampled_itk = resample_sitk_image(image_sitk, attributes=target_attributes, fill_value=float(fill_value))
            image_resampled_npy = GetArrayFromImage(image_resampled_itk)
            batch[name] = torch.from_numpy(image_resampled_npy)

            # debug
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv.npy', batch['suv'].numpy())
            #np.save('/mnt/datasets/ludovic/AutoPET/tmp/cascade.inference.output_found.npy', batch['cascade.inference.output_found'].numpy())
        
        time_resampling_end = time.perf_counter()
        print(f'cascade time loading={time_loading_end - time_loading_start}, resampling={time_resampling_end - time_resampling_start}')
        return batch


def read_case(path: str) -> Dict:
    with lz4.frame.open(path, mode='rb') as f:
        case_data = pickle.load(f)
    return case_data


def write_case(path: str, case_data) -> None:
    with lz4.frame.open(path, mode='wb') as f:
        pickle.dump(case_data, f)


class TransformCascadedPredictionCached:
    """
    Calculate a higher stage (i.e., larger FoV) in a cascaded model. Larger FoV results are
    resampled to the current FoV.

    The results may be slow to be calculated, so they are stored on the local file system.
    """
    def __init__(
            self, 
            model: nn.Module, 
            inference_fn, 
            target_spacing_xyz: Tuple[ float, float, float],
            root_cache_files: Optional[str],
            cascade_name: str = 'cascade',
            device: Optional[torch.device] = None) -> None:
        self.model = model
        self.inference_fn = inference_fn
        self.target_spacing_xyz = target_spacing_xyz
        self.device = device
        self.cascade_name = cascade_name

        def hash_parameters(model):
            # just hash a single parameter. It is
            # very unlikely trainable parameters
            # are identical from training to training
            return str(next(self.model.parameters()).detach().cpu())[:20]

        def hash_model(model):
            return str(self.model)

        self.hash_model = hash_model(self.model)
        self.hash_model_parameter = hash_parameters(model)
        self.root_cache_files = root_cache_files
        os.makedirs(root_cache_files, exist_ok=True)


    def __call__(self, batch: Batch) -> Batch:
        current_spacing_xyz = batch['original_spacing']

        if self.device is not None:
            device = trw.train.get_device(self.model)
            if device != self.device:
                print(f'Moving model to device={self.device}')
                self.model = self.model.to(self.device)

        one_image_name = [n for n, v in batch.items() if isinstance(v, torch.Tensor) and len(v.shape) == 3]

        cache_invalid = True
        cached_case = None
        case_name = batch['case_name']
        cache_location = os.path.join(self.root_cache_files, case_name)
        if self.root_cache_files is not None:
            if os.path.exists(cache_location):
                cached_case = read_case(cache_location)
                if cached_case['hash_model'] == self.hash_model and cached_case['hash_model_parameter'] == self.hash_model_parameter:
                    # the cache is up to date, no need for 
                    # recalculation
                    cache_invalid = False

        if cache_invalid:
            print('cache is invalid!')
            # resample the images to expected spacing
            time_resampling_start = time.perf_counter()
            images = {n: v for n, v in batch.items() if isinstance(v, torch.Tensor) and len(v.shape) == 3}
            resampled_volumes = {}
            for name, image in images.items():
                image_sitk = make_sitk_image(
                    image.numpy(), 
                    origin_xyz=np.zeros([3]), 
                    spacing_xyz=tuple(current_spacing_xyz)
                )
                fill_value = image.min()
                # remember SITK operates in XYZ not ZYX
                target_attributes = get_sitk_image_attributes(image_sitk)
                target_attributes['spacing'] = self.target_spacing_xyz
                target_attributes['shape'] = (image.shape[::-1] * current_spacing_xyz / np.asarray(self.target_spacing_xyz)).round().astype(np.int32)

                image_resampled_itk = resample_sitk_image(image_sitk, attributes=target_attributes, fill_value=float(fill_value))
                image_resampled_npy = GetArrayFromImage(image_resampled_itk)
                resampled_volumes[name] = torch.from_numpy(image_resampled_npy)
            time_resampling_end = time.perf_counter()
            print('time_resampling=', time_resampling_end - time_resampling_start)
            
            # now apply the inference
            time_inference_start = time.perf_counter()
            inference = self.inference_fn(resampled_volumes, self.model)
            time_inference_end = time.perf_counter()
            print('time_inference', time_inference_end - time_inference_start)

            # export the updated cache
            cached_case = {
                'hash_model': self.hash_model,
                'hash_model_parameter': self.hash_model_parameter,
                'inference.output_found': inference.output_found.detach().cpu()
            }

            # resample to original shape
            time_resampling_result_start = time.perf_counter()
            images = {n: v for n, v in cached_case.items() if isinstance(v, torch.Tensor) and len(v.shape) >= 3}
            for image_name, image in images.items():
                image_sitk = make_sitk_image(
                    image.numpy(), 
                    origin_xyz=np.zeros([3]), 
                    spacing_xyz=tuple(self.target_spacing_xyz)
                )
                target_attributes = make_sitk_image_attributes(
                    shape_xyz=batch[one_image_name[0]].shape[::-1],
                    spacing_xyz=current_spacing_xyz
                )

                image_resampled_itk = resample_sitk_image(image_sitk, attributes=target_attributes, fill_value=0)
                image_resampled_npy = GetArrayFromImage(image_resampled_itk)
                cached_case[image_name] = torch.from_numpy(image_resampled_npy)
            time_resampling_result_end = time.perf_counter()
            print('time_resampling_result=', time_resampling_result_end - time_resampling_result_start)

            write_case(cache_location, cached_case)

            # debug only here
            """
            if batch['seg'].max() > 0:
                np.save('/mnt/datasets/ludovic/AutoPET/tmp/seg.npy', batch['seg'].numpy())
                np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv.npy', batch['suv'].numpy())
                np.save('/mnt/datasets/ludovic/AutoPET/tmp/seg_found.npy', cached_case['inference.output_found'].numpy())
            """

        for name, value in cached_case.items():
            batch[self.cascade_name + '_' + name] = value
        return batch