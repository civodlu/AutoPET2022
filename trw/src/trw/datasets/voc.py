import os
import functools
import collections
from typing import Optional, List

from PIL import Image
import torchvision
from ..transforms import stack

from ..train import SequenceArray, SamplerRandom, SamplerSequential
from ..transforms import criteria_feature_name, TransformCompose, TransformResize, TransformNormalizeIntensity

import xml.etree.ElementTree as ET

from ..basic_typing import Datasets
from ..transforms import Transform
from typing_extensions import Literal

from .utils import pic_to_tensor, get_data_root
import numpy as np
import torch


def _load_image_and_mask(batch, transform, normalize_0_1=True):
    images = []
    masks = []
    for image_path, mask_path in zip(batch['images'], batch['masks']):
        image = pic_to_tensor(Image.open(image_path).convert('RGB'))
        if normalize_0_1:
            image = image.float() / 255.0

        images.append(image)

        mask = pic_to_tensor(Image.open(mask_path))
        masks.append(mask)

    batch = {
        'images': stack(images),
        'masks': stack(masks),
    }

    if transform is not None:
        batch = transform(batch)

    return batch


def _parse_voc_xml(node):
    """
    Extracted from torchvision
    """
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(_parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


OBJECT_CLASS_MAPPING = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


def _load_image_and_bb(batch, transform, normalize_0_1=True):
    images = []
    annotations = []
    sizes_cyx = []
    object_class_by_image = []
    object_bb_yx_by_image = []
    label_difficulty_by_image = []
    image_paths = []

    for image_path, annotation_path in zip(batch['images'], batch['annotations']):
        image_paths.append(image_path)
        image = pic_to_tensor(Image.open(image_path).convert('RGB'))
        if normalize_0_1:
            image = image.float() / 255.0
        images.append(image)

        annotation = _parse_voc_xml(ET.parse(annotation_path).getroot())['annotation']
        annotations.append(annotation)

        s = annotation['size']
        sizes_cyx.append((
            int(s['depth']),
            int(s['height']),
            int(s['width'])))

        o_classes = []
        o_bb = []
        o_difficult = []
        for o in annotation['object']:
            o_classes.append(OBJECT_CLASS_MAPPING[o['name']])
            box = o['bndbox']
            o_difficult.append(int(o['difficult']))
            o_bb.append([
                float(box['ymin']),
                float(box['xmin']),
                float(box['ymax']),
                float(box['xmax'])])
        object_class_by_image.append(torch.from_numpy(np.asarray(o_classes, dtype=np.int64)))
        label_difficulty_by_image.append(torch.tensor(o_difficult, dtype=torch.long))

        # typically handled on CPU, so keep it as numpy
        object_bb_yx_by_image.append(np.asarray(o_bb, dtype=np.float32))

    image_scale = np.ones([len(images)], dtype=np.float32)

    batch = {
        'image_path': image_paths,
        'sample_uid': batch['sample_uid'],
        'images': images,
        'image_scale': image_scale,
        'annotations': annotations,
        'sizes_cyx': sizes_cyx,
        'object_class_by_image': object_class_by_image,
        'label_difficulty_by_image': label_difficulty_by_image,
        'object_bb_yx_by_image': object_bb_yx_by_image
    }

    if transform is not None:
        batch = transform(batch)

    return batch


def default_voc_transforms():
    criteria_images = functools.partial(criteria_feature_name, feature_names=['images'])
    return TransformCompose([
        TransformResize(size=[250, 250]),
        #TransformRandomCropPad(feature_names=['images', 'masks'], padding=None, shape=[3, 224, 224]),
        TransformNormalizeIntensity(criteria_fn=criteria_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_voc_segmentation_dataset(
        batch_size: int = 40,
        root: Optional[str] = None,
        transform_train: Optional[List[Transform]] = default_voc_transforms(),
        transform_valid: Optional[List[Transform]] = None,
        nb_workers: int = 2,
        year: Literal['2007', '2012'] = '2012') -> Datasets:
    """
    Create the VOC segmentation dataset

    Args:
        batch_size: the number of samples per batch
        root: the root of the dataset
        transform_train: the transform to apply on each batch of data of the training data
        transform_valid: the transform to apply on each batch of data of the validation data
        nb_workers: the number of worker process to pre-process the batches
        year: the version of the dataset

    Returns:
        a datasets with dataset `voc2012` and splits `train`, `valid`.
    """
    root = get_data_root(root)
    path = os.path.join(root, f'VOC{year}')

    download = False
    try:
        # test if we have access to the data. If this fails, it means we need to download it!
        torchvision.datasets.VOCSegmentation(root=path, image_set='val', transform=None, download=False, year=year)
    except:
        download = True

    # train split

    # here batch_size = 1 since the images do not all have the same size, so we need to process them
    # independently. The `transform` should normalize the size (resize, central_crop) so that
    # the images can be batched subsequently
    train_dataset = torchvision.datasets.VOCSegmentation(
        root=path, image_set='train', transform=None, download=download, year=year)
    train_sequence = SequenceArray({
        'images': train_dataset.images,
        'masks': train_dataset.masks,
    }, SamplerRandom(batch_size=1))
    train_sequence = train_sequence.map(functools.partial(_load_image_and_mask, transform=transform_train), nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)
    if batch_size != 1:
        train_sequence = train_sequence.batch(batch_size)

    # valid split
    valid_dataset = torchvision.datasets.VOCSegmentation(
        root=path, image_set='val', transform=None, download=download, year=year)
    valid_sequence = SequenceArray({
        'images': valid_dataset.images,
        'masks': valid_dataset.masks,
    }, SamplerSequential(batch_size=1))
    valid_sequence = valid_sequence.map(functools.partial(_load_image_and_mask, transform=transform_valid), nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)
    if batch_size != 1:
        valid_sequence = valid_sequence.batch(batch_size)

    return {
        f'voc{year}_seg': collections.OrderedDict([
            ('train', train_sequence.collate()),
            ('valid', valid_sequence.collate()),
        ])
    }


def create_voc_detection_dataset(
        root: str = None,
        transform_train: Optional[List[Transform]] = None,
        transform_valid: Optional[List[Transform]] = None,
        nb_workers: int = 2,
        batch_size: int = 1,
        data_subsampling_fraction_train: float = 1.0,
        data_subsampling_fraction_valid: float = 1.0,
        train_split: str = 'train',
        valid_split: str = 'val',
        year: Literal['2007', '2012'] = '2007') -> Datasets:
    """
    PASCAL VOC detection challenge

    Notes:
        - Batch size is always `1` since we need to sample from the image various anchors,
          locations depending on the task (so each sample should be post-processed by a custom
          transform)
    """
    root = get_data_root(root)

    #path = os.path.join(root, f'VOC{year}')  # TODO
    path = root

    download = False
    try:
        # test if we have access to the data. If this fails, it means we need to download it!
        torchvision.datasets.VOCDetection(
            root=path,
            image_set=train_split,
            transform=None,
            download=False,
            year=year
        )
    except:
        download = True

    train_dataset = torchvision.datasets.VOCDetection(
        root=path,
        image_set=train_split,
        transform=None,
        download=download,
        year=year)

    if data_subsampling_fraction_train < 1.0:
        # resample the data if required
        nb_train = int(len(train_dataset.images) * data_subsampling_fraction_train)
        indices = np.random.choice(len(train_dataset.images), nb_train, replace=False)
        train_dataset.images = np.asarray(train_dataset.images)[indices].tolist()
        train_dataset.annotations = np.asarray(train_dataset.annotations)[indices].tolist()

    train_sequence = SequenceArray({
        'images': train_dataset.images,
        'annotations': train_dataset.annotations
    }, SamplerRandom(batch_size=batch_size))
    train_sequence = train_sequence.map(functools.partial(_load_image_and_bb, transform=transform_train),
                                        nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)

    # valid split
    valid_dataset = torchvision.datasets.VOCDetection(
        root=path,
        image_set=valid_split,
        transform=None,
        download=download,
        year=year)

    if data_subsampling_fraction_valid < 1.0:
        # resample the data if required
        nb_valid = int(len(valid_dataset.images) * data_subsampling_fraction_valid)
        indices = np.random.choice(len(valid_dataset.images), nb_valid, replace=False)
        valid_dataset.images = np.asarray(valid_dataset.images)[indices].tolist()
        valid_dataset.annotations = np.asarray(valid_dataset.annotations)[indices].tolist()

    valid_sequence = SequenceArray({
        'images': valid_dataset.images,
        'annotations': valid_dataset.annotations,
    }, SamplerSequential(batch_size=batch_size))
    valid_sequence = valid_sequence.map(functools.partial(_load_image_and_bb, transform=transform_valid),
                                        nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)

    return {
        f'voc{year}_detect': collections.OrderedDict([
            ('train', train_sequence),
            ('valid', valid_sequence),
        ])
    }
