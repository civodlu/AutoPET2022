import collections
from unittest import TestCase
import trw.train
import trw.transforms
import numpy as np
import torch
import functools
from trw.transforms import TransformSqueeze
from trw.transforms.transforms_unsqueeze import TransformUnsqueeze
import trw.utils


class TransformRecorder(trw.transforms.Transform):
    def __init__(self, kvp, tfm_id):
        self.kvp = kvp
        self.tfm_id = tfm_id

    def __call__(self, batch):
        self.kvp[self.tfm_id] += 1
        return batch


class TestTransform(TestCase):
    def test_batch_pad_constant_numpy(self):
        d = np.asarray([[4], [5], [6]], dtype=int)
        d_transformed = trw.utils.batch_pad_numpy(d, [2], mode='constant', constant_value=9)
        self.assertTrue(d_transformed.shape == (3, 5))
        assert (d_transformed[0] == [9, 9, 4, 9, 9]).all()
        assert (d_transformed[1] == [9, 9, 5, 9, 9]).all()
        assert (d_transformed[2] == [9, 9, 6, 9, 9]).all()

    def test_batch_pad_constant_torch(self):
        d = np.asarray([[4], [5], [6]], dtype=int)
        d = torch.from_numpy(d)
        d_transformed = trw.utils.batch_pad_torch(d, [2], mode='constant', constant_value=9)
        d_transformed = d_transformed.data.numpy()
        self.assertTrue(d_transformed.shape == (3, 5))
        assert (d_transformed[0] == [9, 9, 4, 9, 9]).all()
        assert (d_transformed[1] == [9, 9, 5, 9, 9]).all()
        assert (d_transformed[2] == [9, 9, 6, 9, 9]).all()

    def test_batch_pad_symmetric_numpy(self):
        d = np.asarray([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=int)
        d_transformed = trw.utils.batch_pad_numpy(d, [2], mode='symmetric')
        self.assertTrue(d_transformed.shape == (3, 7))

    def test_batch_pad_edge_torch(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]
        d = np.asarray([i1, i2], dtype=float)
        d = d.reshape((2, 1, 3, 3))
        d = torch.from_numpy(d)
        d_transformed = trw.utils.batch_pad_torch(d, [0, 2, 3], mode='edge')
        d_transformed = d_transformed.data.numpy()
        self.assertTrue(d_transformed.shape == (2, 1, 7, 9))

    def test_batch_pad_replicate_numpy(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]
        d = np.asarray([i1, i2], dtype=float)
        d = d.reshape((2, 1, 3, 3))
        d_transformed = trw.utils.batch_pad_numpy(d, [0, 2, 3], mode='edge')
        self.assertTrue(d_transformed.shape == (2, 1, 7, 9))

    def test_batch_pad_constant_2d_numpy(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]

        d = np.asarray([i1, i2], dtype=int)
        d_transformed = trw.utils.batch_pad_numpy(d, [2, 3], mode='constant')
        self.assertTrue(d_transformed.shape == (2, 7, 9))

    def test_batch_pad_constant_2d_torch(self):
        i1 = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        i2 = [[40, 41, 42], [50, 51, 52], [60, 61, 62]]
        d = np.asarray([i1, i2], dtype=int)
        d = torch.from_numpy(d)

        d_transformed = trw.utils.batch_pad_torch(d, [2, 3], mode='constant')
        d_transformed = d_transformed.data.numpy()
        self.assertTrue(d_transformed.shape == (2, 7, 9))

    def test_random_crop_numpy(self):
        d = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
        d_transformed = trw.transforms.transform_batch_random_crop(d, [2])
        self.assertTrue((d_transformed[0] == [1, 2]).all() or (d_transformed[0] == [2, 3]).all())
        self.assertTrue((d_transformed[1] == [4, 5]).all() or (d_transformed[1] == [5, 6]).all())
        self.assertTrue((d_transformed[2] == [7, 8]).all() or (d_transformed[2] == [8, 9]).all())

    def test_random_crop_torch(self):
        d = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=int)
        d = torch.from_numpy(d)
        d_transformed = trw.transforms.transform_batch_random_crop(d, [2])
        d_transformed = d_transformed.data.numpy()
        self.assertTrue((d_transformed[0] == [1, 2]).all() or (d_transformed[0] == [2, 3]).all())
        self.assertTrue((d_transformed[1] == [4, 5]).all() or (d_transformed[1] == [5, 6]).all())
        self.assertTrue((d_transformed[2] == [7, 8]).all() or (d_transformed[2] == [8, 9]).all())

    def test_random_crop_padd_transform_numpy(self):
        size = [1, 46, 63]
        d = np.zeros([6000] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1

        transform = trw.transforms.TransformRandomCropPad(padding=[0, 8, 8])
        batch = transform({'d': d})

        assert batch['d'].shape == (6000, 1, 46, 63)
        d_summed = np.sum(batch['d'], axis=0).squeeze()
        ys, xs = np.where(d_summed > 0)

        # we have set one's at the center of a 2D image, test the maximum and
        # minimum displacement
        self.assertTrue(min(ys) == size[1] // 2 - 8)
        self.assertTrue(max(ys) == size[1] // 2 + 8)

        self.assertTrue(min(xs) == size[2] // 2 - 8)
        self.assertTrue(max(xs) == size[2] // 2 + 8)

    def test_random_crop_padd_transform_torch(self):
        size = [1, 46, 63]
        nb = 6000

        d = np.zeros([nb] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1.0
        d = torch.from_numpy(d)

        transform = trw.transforms.TransformRandomCropPad(padding=[0, 8, 8])
        batch = transform({'d': d})

        d_transformed = batch['d'].data.numpy()

        assert d_transformed.shape == (nb, size[0], size[1], size[2])
        d_summed = np.sum(d_transformed, axis=0).squeeze()
        ys, xs = np.where(d_summed > 0)

        # we have set one's at the center of a 2D image, test the maximum and
        # minimum displacement
        self.assertTrue(min(ys) == size[1] // 2 - 8)
        self.assertTrue(max(ys) == size[1] // 2 + 8)

        self.assertTrue(min(xs) == size[2] // 2 - 8)
        self.assertTrue(max(xs) == size[2] // 2 + 8)

    def test_random_crop_no_padding(self):
        size = [1, 31, 63]
        d = np.zeros([1000] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1.0
        d = torch.from_numpy(d)

        transform = trw.transforms.TransformRandomCropPad(padding=None, shape=[1, 16, 32])
        batch = transform({'d': d})

        d_transformed = batch['d'].data.numpy()

        assert d_transformed.shape == (1000, 1, 16, 32)

    def test_random_crop_resize(self):
        size = [1, 31, 63]
        d = np.zeros([1000] + size, dtype=np.float)
        d[:, size[0] // 2, size[1] // 2, size[2] // 2] = 1.0
        d = torch.from_numpy(d)

        transform = trw.transforms.TransformRandomCropResize([16, 32])
        batch = transform({'d': d})

        d_transformed = batch['d'].data.numpy()

        assert d_transformed.shape == (1000, 1, 31, 63)

    def test_transform_base_criteria(self):
        # filter by name
        batch = {
            'test_1': 0,
            'test_2': 42,
        }

        criteria_fn = functools.partial(trw.transforms.criteria_feature_name, feature_names=['test_2'])

        def transform_fn(features_to_transform, batch):
            for name, value in batch.items():
                if name in features_to_transform:
                    batch[name] = 43
            return batch

        transformer = trw.transforms.TransformBatchWithCriteria(criteria_fn=criteria_fn, transform_fn=transform_fn)
        transformed_batch = transformer(batch)
        assert transformed_batch['test_1'] == 0
        assert transformed_batch['test_2'] == 43

    def test_transform_random_flip_numpy(self):
        batch = {
            'images': np.asarray([
                [1, 2, 3],
                [4, 5, 6]
            ])
        }

        criteria_fn = functools.partial(trw.transforms.criteria_feature_name, feature_names=['images'])
        transformer = trw.transforms.TransformRandomFlip(criteria_fn=criteria_fn, axis=1, flip_probability=1.0)
        transformed_batch = transformer(batch)

        image_fliped = transformed_batch['images']
        assert len(image_fliped) == 2
        assert image_fliped.shape == (2, 3)

        assert image_fliped[0, 0] == 3
        assert image_fliped[0, 1] == 2
        assert image_fliped[0, 2] == 1

        assert image_fliped[1, 0] == 6
        assert image_fliped[1, 1] == 5
        assert image_fliped[1, 2] == 4

        # make sure the original images are NOT flipped!
        image = batch['images']
        assert image[0, 0] == 1
        assert image[0, 1] == 2
        assert image[0, 2] == 3

    def test_transform_random_flip_torch(self):
        batch = {
            'images': torch.from_numpy(np.asarray([
                [1, 2, 3],
                [4, 5, 6]
            ]))
        }

        criteria_fn = functools.partial(trw.transforms.criteria_feature_name, feature_names=['images'])
        transformer = trw.transforms.TransformRandomFlip(criteria_fn=criteria_fn, axis=1, flip_probability=1.0)
        transformed_batch = transformer(batch)

        image_fliped = transformed_batch['images']
        assert len(image_fliped) == 2
        assert image_fliped.shape == (2, 3)

        assert image_fliped[0, 0] == 3
        assert image_fliped[0, 1] == 2
        assert image_fliped[0, 2] == 1

        assert image_fliped[1, 0] == 6
        assert image_fliped[1, 1] == 5
        assert image_fliped[1, 2] == 4

        # make sure the original images are NOT flipped!
        image = batch['images']
        assert image[0, 0] == 1
        assert image[0, 1] == 2
        assert image[0, 2] == 3

    def test_cutout_numpy(self):
        batch = {
            'images': np.ones([50, 3, 64, 128], dtype=np.uint8) * 255
        }
        transformer = trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 32))
        transformed_batch = transformer(batch)
        assert np.min(batch['images']) == 255, 'original image was modified!'

        assert np.min(transformed_batch['images']) == 0, 'transformed image was NOT modified!'
        for i in transformed_batch['images']:
            nb_0 = np.where(i == 0)
            assert len(nb_0[0]) == 3 * 16 * 32

    def test_cutout_torch(self):
        batch = {
            'images': torch.ones([50, 3, 64, 128], dtype=torch.uint8) * 255
        }
        transformer = trw.transforms.TransformRandomCutout(cutout_size=(3, 16, 32))
        transformed_batch = transformer(batch)
        assert torch.min(batch['images']) == 255, 'original image was modified!'

        assert torch.min(transformed_batch['images']) == 0, 'transformed image was NOT modified!'
        for i in transformed_batch['images']:
            nb_0 = np.where(i.numpy() == 0)
            assert len(nb_0[0]) == 3 * 16 * 32

    def test_cutout_size_functor(self):
        batch = {
            'images': torch.ones([50, 3, 64, 128], dtype=torch.uint8) * 255
        }

        size_functor_called = 0

        def cutout_size_fn():
            nonlocal size_functor_called
            s = trw.transforms.cutout_random_size([3, 5, 5], [3, 10, 10])
            assert s[0] == 3
            assert s[1] >= 5
            assert s[2] >= 5

            assert s[1] <= 10
            assert s[2] <= 10
            size_functor_called += 1
            return s

        transformer = trw.transforms.TransformRandomCutout(
            cutout_size=cutout_size_fn,
            cutout_value_fn=trw.transforms.cutout_random_ui8_torch)
        _ = transformer(batch)
        assert size_functor_called == 50

    def test_random_crop_pad_joint(self):
        batch = {
            'images': torch.zeros([50, 3, 64, 128], dtype=torch.int64),
            'segmentations': torch.zeros([50, 1, 64, 128], dtype=torch.float32),
            'something_else': 42,
        }

        batch['images'][:, :, 32, 64] = 42
        batch['segmentations'][:, :, 32, 64] = 42

        transformer = trw.transforms.TransformRandomCropPad(
            criteria_fn=lambda batch, names: ['images', 'segmentations'],
            padding=[0, 16, 16],
            mode='constant')
        transformed_batch = transformer(batch)

        indices_images_42 = np.where(transformed_batch['images'].numpy()[:, 0, :, :] == 42)
        indices_segmentations_42 = np.where(transformed_batch['segmentations'].numpy()[:, 0, :, :] == 42)
        assert (indices_segmentations_42[2] == indices_images_42[2]).all()
        assert (indices_segmentations_42[1] == indices_images_42[1]).all()

    def test_random_flipped_joint(self):
        batch = {
            'images': torch.randint(high=42, size=[50, 3, 64, 128], dtype=torch.int64),
        }
        batch['segmentation'] = batch['images'].float()

        transformer = trw.transforms.TransformRandomFlip(criteria_fn=lambda _: ['images', 'segmentation'], axis=2)
        transformed_batch = transformer(batch)

        images = transformed_batch['images'].float()
        segmentations = transformed_batch['segmentation'].float()
        assert (images == segmentations).all()

    def test_random_resize_torch(self):
        batch = {
            'images': torch.randint(high=42, size=[50, 3, 16, 32], dtype=torch.int64),
        }

        transformer = trw.transforms.TransformResize(size=[32, 64])
        transformed_batch = transformer(batch)

        images = transformed_batch['images']
        assert images.shape == (50, 3, 32, 64)
        assert np.average(batch['images'].numpy()) == np.average(transformed_batch['images'].numpy())

    def test_random_resize_numpy(self):
        batch = {
            'images': np.random.randint(low=0, high=42, size=[50, 3, 16, 32], dtype=np.int64),
        }

        transformer = trw.transforms.TransformResize(size=[32, 64], mode='nearest')
        transformed_batch = transformer(batch)

        images = transformed_batch['images']
        assert images.shape == (50, 3, 32, 64)
        assert np.average(batch['images']) == np.average(transformed_batch['images'])

    def test_normalize_numpy(self):
        images = np.random.randint(low=0, high=42, size=[10, 3, 5, 6], dtype=np.int64)
        images[:, 1] *= 2
        images[:, 2] *= 3

        batch = {
            'images': images,
        }

        mean = np.mean(images, axis=(0, 2, 3))
        std = np.std(images, axis=(0, 2, 3))

        transformer = trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
        transformed_batch = transformer(batch)
        normalized_images = transformed_batch['images']

        mean_normalized = np.mean(normalized_images, axis=(0, 2, 3))
        std_normalized = np.std(normalized_images, axis=(0, 2, 3))

        assert abs(np.average(mean_normalized)) < 0.1
        assert abs(np.average(std_normalized) - 1) < 0.1

    def test_normalize_torch(self):
        images = torch.randint(high=42, size=[10, 3, 5, 6], dtype=torch.float32)
        images[:, 1] *= 2
        images[:, 2] *= 3

        batch = {
            'images': images,
        }

        mean = np.mean(images.numpy(), axis=(0, 2, 3), dtype=np.float32)
        std = np.std(images.numpy(), axis=(0, 2, 3), dtype=np.float32)

        transformer = trw.transforms.TransformNormalizeIntensity(mean=mean, std=std)
        transformed_batch = transformer(batch)
        normalized_images = transformed_batch['images']

        mean_normalized = np.mean(normalized_images.numpy(), axis=(0, 2, 3))
        std_normalized = np.std(normalized_images.numpy(), axis=(0, 2, 3))

        assert abs(np.average(mean_normalized)) < 0.1
        assert abs(np.average(std_normalized) - 1) < 0.1

    def test_transform_compose(self):
        batch = {
            'images': torch.randint(high=42, size=[10, 3, 5, 6], dtype=torch.int64).float()
        }

        transforms = [
            trw.transforms.TransformNormalizeIntensity(mean=[np.float32(10), np.float32(10), np.float32(10)], std=[np.float32(1), np.float32(1), np.float32(1)]),
            trw.transforms.TransformNormalizeIntensity(mean=[np.float32(100), np.float32(100), np.float32(100)], std=[np.float32(1), np.float32(1), np.float32(1)]),
        ]

        transformer = trw.transforms.TransformCompose(transforms)
        transformed_batch = transformer(batch)

        max_error = torch.max(torch.abs(batch['images'] - transformed_batch['images'] - 110))
        assert float(max_error) < 1e-5

    def test_transform_random_flip_joint(self):
        np.random.seed(0)

        batch = {
            'images': np.asarray([
                [1, 2, 3],
                [4, 5, 6],
            ]),
            'images2': np.asarray([
                [1, 2, 3],
                [4, 5, 6],
            ])
        }

        transformer = trw.transforms.TransformRandomFlip(
            criteria_fn=lambda _: ['images', 'images2'],
            axis=1,
            flip_probability=0.5)
        transformed_batch = transformer(batch)

        image_fliped = transformed_batch['images']
        image_fliped2 = transformed_batch['images2']
        assert len(image_fliped) == 2
        assert image_fliped.shape == (2, 3)

        assert (image_fliped == image_fliped2).all()

        # make sure the original images are NOT flipped!
        image = batch['images']
        assert image[0, 0] == 1
        assert image[0, 1] == 2
        assert image[0, 2] == 3

    def test_batch_crop(self):
        i = np.random.randint(0, 100, [16, 20, 24, 28])

        c = trw.transforms.batch_crop(i, [10, 11, 12], [12, 15, 20])
        assert c.shape == (16, 2, 4, 8)
        assert (c == i[:, 10:12, 11:15, 12:20]).all()

    def test_cast_numpy(self):
        batch = {
            'float': np.zeros([10], dtype=np.long),
            'long': np.zeros([10], dtype=np.long),
            'byte': np.zeros([10], dtype=np.long),
        }

        transforms = [
            trw.transforms.TransformCast(['float'], 'float'),
            trw.transforms.TransformCast(['long'], 'long'),
            trw.transforms.TransformCast(['byte'], 'byte'),
        ]
        tfm = trw.transforms.TransformCompose(transforms)

        batch_tfm = tfm(batch)
        assert batch_tfm['float'].dtype == np.float32
        assert batch_tfm['long'].dtype == np.long
        assert batch_tfm['byte'].dtype == np.byte

    def test_cast_torch(self):
        batch = {
            'float': torch.zeros([10], dtype=torch.long),
            'long': torch.zeros([10], dtype=torch.float),
            'byte': torch.zeros([10], dtype=torch.long),
        }

        transforms = [
            trw.transforms.TransformCast(['float'], 'float'),
            trw.transforms.TransformCast(['long'], 'long'),
            trw.transforms.TransformCast(['byte'], 'byte'),
        ]
        tfm = trw.transforms.TransformCompose(transforms)

        batch_tfm = tfm(batch)
        assert batch_tfm['float'].dtype == torch.float32
        assert batch_tfm['long'].dtype == torch.long
        assert batch_tfm['byte'].dtype == torch.int8

    def test_one_of(self):
        np.random.seed(0)
        nb_samples = 10000
        split = {
            'float': torch.zeros([nb_samples], dtype=torch.long),
        }

        kvp = collections.defaultdict(lambda: 0)

        transforms = [
            TransformRecorder(kvp, 0),
            TransformRecorder(kvp, 1),
            TransformRecorder(kvp, 2),
        ]
        tfm = trw.transforms.TransformOneOf(transforms)
        for b in trw.train.SequenceArray(split):
            _ = tfm(b)

        nb_transforms_applied = sum(kvp.values())

        assert nb_transforms_applied == nb_samples
        tolerance = 0.01 * nb_samples
        for tfm, tfm_count in kvp.items():
            deviation = abs(tfm_count - nb_samples / len(transforms))
            assert deviation < tolerance, f'deviation={deviation}, tolerance={tolerance}'

    def test_transform_squeeze(self):
        nb_samples = 10
        split = {
            'float_torch': torch.zeros([nb_samples, 4, 1, 5, 6], dtype=torch.float32),
            'float_np': np.zeros([nb_samples, 4, 1, 5, 6], dtype=np.float32),
            'str': 'a string',
            'number': 4.0
        }

        tfm = TransformSqueeze(axis=2)

        split_tfm = tfm(split) 
        assert len(split_tfm) == 4
        assert split_tfm['float_torch'].shape == (nb_samples, 4, 5, 6)
        assert split_tfm['float_np'].shape == (nb_samples, 4, 5, 6)

    def test_transform_unsqueeze(self):
        nb_samples = 10
        split = {
            'float_torch': torch.zeros([nb_samples, 4, 5, 6], dtype=torch.float32),
            'float_np': np.zeros([nb_samples, 4, 5, 6], dtype=np.float32),
            'str': 'a string',
            'number': 4.0
        }

        tfm = TransformUnsqueeze(axis=2)

        split_tfm = tfm(split) 
        assert len(split_tfm) == 4
        assert split_tfm['float_torch'].shape == (nb_samples, 4, 1, 5, 6)
        assert split_tfm['float_np'].shape == (nb_samples, 4, 1, 5, 6)

    def test_transform_to_device(self):
        device = 'cpu'
        if torch.cuda.device_count() > 0:
            device = 'cuda:0'
        device = torch.device(device)

        batch = {
            'test': 'should not be moved!',
            'test2': torch.zeros([5, 5], dtype=torch.float32, device=torch.device('cpu'))
        }

        tfm = trw.transforms.TransformMoveToDevice(device=device)
        batch_tfm = tfm(batch)
        assert len(batch_tfm) == len(batch)
        assert batch_tfm['test'] is batch['test']
        assert batch_tfm['test2'].device == device