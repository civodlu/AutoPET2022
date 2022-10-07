import functools
from unittest import TestCase
import trw
import numpy as np
import torch
from trw.transforms import resample_3d, SpatialInfo, random_fixed_geometry_within_geometries, \
    affine_transformation_translation
from trw.transforms.affine import affine_transformation_rotation_3d_x, affine_transformation_scale, \
    apply_homogeneous_affine_transform, apply_homogeneous_affine_transform_zyx
from trw.transforms.resample import resample_spatial_info
from trw.utils import torch_requires


class TestTransformResample(TestCase):
    # align_corners not supported in pytorch 1.0-1.2, `resample_spatial_info` will not be accurate
    # for these versions! Disable the test
    @torch_requires(min_version='1.3', silent_fail=True)
    def test_resample_numpy_copy_subvolume(self):
        """
        Simply copy a sub-volume
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (2, 3, 4), (4, 5, 6), (1, 1, 1), align_corners=False)
        t_expected = t[2:4, 3:5, 4:6]
        assert np.abs(t_r - t_expected).max() < 1e-4

    # align_corners not supported in pytorch 1.0-1.2, `resample_spatial_info` will not be accurate
    # for these versions! Disable the test
    @torch_requires(min_version='1.3', silent_fail=True)
    def test_resample_numpy_with_background(self):
        """
        Simply copy a sub-volume with background voxels
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (2, 3, 4), (4, 5, 20), (1, 1, 1), align_corners=False)
        t_r_valid = t_r[:, :, :2]
        t_expected = t[2:4, 3:5, 4:6]
        assert np.abs((t_r_valid - t_expected)).max() < 1e-4

        # outside voxel should be background value
        t_r_background = t_r[:, :, 4:]
        assert (t_r_background == 0).all()

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_resample_numpy_id(self):
        """
        identity
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (0, 0, 0), (10, 9, 8), (1, 1, 1), interpolation_mode='nearest')
        assert np.abs(t - t_r).max() < 1e-4

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_resample_numpy_interpolate_x(self):
        """
        Interpolate in x axis
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8)).numpy()
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (0, 0, 0), (10, 9, 8), (1, 1, 0.49999), interpolation_mode='nearest')
        assert t_r.shape == (10, 9, 8 * 2)
        for z in range(t_r.shape[0]):
            for y in range(t_r.shape[1]):
                for x in range(t_r.shape[2] - 1):  # avoid the background value at the end
                    v = t_r[z, y, x]
                    v_expected = t[z, y, x // 2]
                    assert abs(v - v_expected) < 1e-3

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_resample_torch_id(self):
        """
        identity
        """
        t = torch.arange(10 * 9 * 8).reshape((10, 9, 8))
        t_r = resample_3d(t, (1, 1, 1), (0, 0, 0), (0, 0, 0), (10, 9, 8), (1, 1, 1), interpolation_mode='nearest')
        assert np.abs(t.float() - t_r).max() < 1e-5

    @staticmethod
    def get_spatial_info_generic(batch, name, geometry) -> SpatialInfo:
        return geometry

    @staticmethod
    def get_random_geometry(dict_of_g):
        g = SpatialInfo(
            shape=np.random.randint(1, 8, size=[3]),
            origin=np.random.uniform(-4, 6, size=[3]),
            spacing=np.random.uniform(0.5, 1.5, size=[3]),
        )
        return g

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_transform_resampling_random_dependent(self):
        """
        Use a functors to randomly generate a resampling geometry.

        All volumes MUST use the same geometry.
        """
        batch = {
            'v1': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'v2': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'other': 'other_value'
        }

        get_spatial_info = functools.partial(TestTransformResample.get_spatial_info_generic,
                                             geometry=SpatialInfo(origin=(0, 0, 0), spacing=(1, 1, 1), shape=(10, 9, 8)))

        transform = trw.transforms.TransformResample(
            get_spatial_info_from_batch_name=get_spatial_info,
            resampling_geometry=TestTransformResample.get_random_geometry,
        )

        batch_transformed = transform(batch)
        assert len(batch_transformed) == 3
        assert batch_transformed['other'] == 'other_value'
        assert batch_transformed['v1'].shape == batch_transformed['v2'].shape

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_transform_resampling_random_fixed(self):
        batch = {
            'v1': torch.arange(10 * 9 * 8).reshape((1, 1, 10, 9, 8)),
            'v2': torch.arange(1 * 1 * 1).reshape((1, 1, 1, 1, 1)),
        }

        get_spatial_info = functools.partial(TestTransformResample.get_spatial_info_generic,
                                             geometry=SpatialInfo(origin=(0, 0, 0),
                                                                  spacing=(1, 1, 1),
                                                                  shape=(10, 9, 8)))

        transform = trw.transforms.TransformResample(
            get_spatial_info_from_batch_name=get_spatial_info,
            resampling_geometry=functools.partial(random_fixed_geometry_within_geometries, fixed_geometry_shape=(3, 3, 3), fixed_geometry_spacing=(1, 1, 1)),
        )

        batch_transformed = transform(batch)
        assert len(batch_transformed) == 2
        assert batch_transformed['v1'].shape == batch_transformed['v2'].shape

    def test_spatial_info_pst(self):
        pst = affine_transformation_translation([10, 11, 12]).mm(
              affine_transformation_rotation_3d_x(0.3)).mm(
              affine_transformation_scale([2, 3, 4]))

        si = SpatialInfo(shape=[20, 21, 22], patient_scale_transform=pst)

        assert (si.spacing == np.asarray([4, 3, 2])).all()
        assert (si.origin == np.asarray([12, 11, 10])).all()

    def test_spatial_info_coordinate_mapping(self):
        origin = torch.tensor([10, 11, 12], dtype=torch.float32)
        spacing = torch.tensor([2, 3, 4], dtype=torch.float32)
        pst = affine_transformation_translation(origin).mm(affine_transformation_scale(spacing))
        si = SpatialInfo(shape=[20, 21, 22], patient_scale_transform=pst)

        origin_flipped = torch.flip(origin, (0,))
        spacing_flipped = torch.flip(spacing, (0,))
        si_2 = SpatialInfo(shape=[20, 21, 22], origin=origin_flipped, spacing=spacing_flipped)
        assert (si.patient_scale_transform - si_2.patient_scale_transform).abs().max() < 1e-5

        # index (0, 0, 0) is origin!
        p = si.index_to_position(index_zyx=torch.tensor([0, 0, 0]))
        assert len(p.shape) == 1
        assert p.shape[0] == 3

        assert (p - torch.tensor([12, 11, 10], dtype=torch.float32)).abs().max() < 1e-5
        assert (si.position_to_index(position_zyx=p) - torch.tensor([0, 0, 0], dtype=torch.float32)).abs().max() < 1e-5

        # move in one direction from the origin
        for n in range(20):
            dp = torch.empty(3).uniform_(0, 10).type(torch.float32)
            p = origin_flipped + dp * spacing_flipped
            i = si.position_to_index(position_zyx=p)
            assert (i - dp).abs().max() < 1e-5

            p_back = si.index_to_position(index_zyx=i)
            assert (p - p_back).abs().max() < 1e-5

    @torch_requires(min_version='1.3', silent_fail=True)
    def test_random_volumes(self):
        def fill_volume(v, nb):
            for n in range(nb):
                min_index = torch.tensor([
                    torch.randint(0, v.shape[0] - 5, size=(1,)),
                    torch.randint(0, v.shape[1] - 5, size=(1,)),
                    torch.randint(0, v.shape[2] - 5, size=(1,)),
                ])
                shape = torch.randint(10, 15, size=(3,))
                value = torch.randint(255, size=(1,))

                max_index = min_index + shape
                max_index = trw.utils.clamp_n(max_index, torch.tensor([0, 0, 0]), torch.tensor(v.shape)).squeeze(0)
                trw.utils.sub_tensor(v, min_index, max_index)[:] = value

        np.random.seed(0)
        torch.manual_seed(0)
        nb_points = 10000
        all_avg_errors = []
        for e in range(10):
            tfm = torch.from_numpy(np.asarray([
                [0.95, 0, 0, -0.5],
                [0, 0.9, 0, -5],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ], dtype=np.float32))

            shape_moving_zyx = (42, 30, 40)
            shape_fixed_zyx = (37, 31, 42)

            moving_geometry = SpatialInfo(shape=shape_moving_zyx, origin=(10, 15, 20), spacing=(0.9, 1.1, 1.3))
            fixed_geometry = SpatialInfo(shape=shape_fixed_zyx, origin=(15, 5, 15), spacing=(1.3, 1.4, 1.1))

            moving = torch.zeros(shape_moving_zyx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            fill_volume(moving[0, 0], 5)

            fixed = resample_spatial_info(
                geometry_moving=moving_geometry,
                moving_volume=moving,
                geometry_fixed=fixed_geometry,
                interpolation='nearest',
                tfm=tfm
            )

            error = 0.0
            nb_voxels = 0

            shape_moving_zyx = torch.tensor(shape_fixed_zyx)

            for _ in range(nb_points):
                index_fixed = torch.tensor([
                    torch.randint(shape_fixed_zyx[0], size=(1,)),
                    torch.randint(shape_fixed_zyx[1], size=(1,)),
                    torch.randint(shape_fixed_zyx[2], size=(1,))
                ], dtype=torch.float32)

                index_fixed_i = index_fixed.type(torch.long)

                # transform: index (fixed) -> position in world space
                p = fixed_geometry.index_to_position(index_zyx=index_fixed)

                # apply moving transform
                p = apply_homogeneous_affine_transform_zyx(tfm, p)

                # transform position -> index (moving)
                index_moving = moving_geometry.position_to_index(position_zyx=p)

                index_moving_rounded = index_moving.round().type(torch.long)
                if (index_moving_rounded >= 0).all() and \
                        ((index_moving_rounded - shape_moving_zyx) < 0).all():
                    value_resampled = fixed[0, 0][index_fixed_i[0], index_fixed_i[1], index_fixed_i[2]]
                    if value_resampled <= 2:
                        # discard background values
                        continue
                    value_fixed = moving[0, 0][
                        index_moving_rounded[0], index_moving_rounded[1], index_moving_rounded[2]]
                    error += (value_resampled - value_fixed).abs()
                    nb_voxels += 1

            avg_error = error / nb_voxels
            print('avg=', error / nb_voxels, 'nb_voxels=', nb_voxels)
            # empirical value
            assert avg_error <= .5
            all_avg_errors.append(avg_error)
        assert np.mean(all_avg_errors) < 0.1

    def test_sub_geometry(self):
        pst = affine_transformation_translation([10, 11, 12]).mm(
            affine_transformation_rotation_3d_x(0.3)).mm(
            affine_transformation_scale([2, 3, 4]))

        si = SpatialInfo(shape=[20, 21, 22], patient_scale_transform=pst)
        start_zyx = torch.tensor([5, 6, 7])
        end_zyx = torch.tensor([8, 10, 13])
        si_sub = si.sub_geometry(start_index_zyx=start_zyx, end_index_zyx_inclusive=end_zyx)

        o = si_sub.index_to_position(index_zyx=torch.tensor([0, 0, 0]))
        expected_o = si.index_to_position(index_zyx=start_zyx)
        assert (o - expected_o).abs().max() == 0

        index_e = end_zyx - start_zyx
        pos_e = si_sub.index_to_position(index_zyx=index_e)
        pos_e_expected = si.index_to_position(index_zyx=end_zyx)
        assert (pos_e - pos_e_expected).abs().max() <= 1e-5

    def test_geometry_center(self):
        pst = affine_transformation_translation([10, 11, 12]).mm(
            affine_transformation_rotation_3d_x(0.3)).mm(
            affine_transformation_scale([2, 3, 4]))
        si = SpatialInfo(shape=[3, 5, 7], patient_scale_transform=pst)

        center_mm_zyx = si.center

        center_xyz = (np.asarray(si.shape[::-1]) - 1) / 2
        center_mm_xyz = center_xyz[0] * pst[:3, 0] + \
                        center_xyz[1] * pst[:3, 1] + \
                        center_xyz[2] * pst[:3, 2] + \
                        pst[:3, 3]
        center_mm_zyx_expected = (center_mm_xyz.numpy())[::-1]
        error = ((center_mm_zyx - center_mm_zyx_expected) ** 2).sum()
        assert error < 1e-6

