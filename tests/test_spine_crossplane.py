import numpy as np

from medimage.spine_crossplane import axial_regions, crop_stack
from medimage.spine_data import STARTS
from medimage.spine_multiscale import FEATURE_COUNT, descriptor


def test_axial_reprojection_preserves_anatomical_center_and_rejects_distant_slices():
    geometry = np.zeros((STARTS[-1], 3, 3), dtype=np.float32)
    sagittal = STARTS[1]
    geometry[sagittal] = [[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    axial = STARTS[2] + np.arange(3)
    geometry[axial] = [[[-10., -10., depth], [1., 0., 0.], [0., 1., 0.]] for depth in (5., 10., 15.)]
    valid = np.zeros(len(geometry), dtype=bool)
    valid[np.r_[sagittal, axial]] = True
    crops = np.array([[[sagittal, 4., 10., 40., 40.]] * 3,
                      [[sagittal, 4., 45., 40., 40.]] * 3], dtype=np.float32)
    other, quality = axial_regions(crops, geometry, valid, 64)
    np.testing.assert_array_equal(other[0, :, 0], axial)
    np.testing.assert_allclose(other[0, :, 1:3], [[10., 14.]] * 3)
    np.testing.assert_allclose(other[0, :, 3:], 40.)
    np.testing.assert_array_equal(other[1, :, 0], [-1, -1, -1])
    np.testing.assert_array_equal(quality[:, 0], [1., 0.])


def test_crop_stack_keeps_center_pixels_and_missing_planes():
    image = np.arange(64, dtype=np.uint8)[None, None, :].repeat(64, axis=1)
    coords = np.array([[-1., 0., 0., 0., 0.], [0., 31.5, 31.5, 64., 64.], [-1., 0., 0., 0., 0.]])
    stack = crop_stack(image, coords)
    np.testing.assert_array_equal(stack[1], image[0])
    assert not stack[[0, 2]].any()


def test_multiscale_features_cover_all_targets_with_missing_regions():
    volume = np.full((1, 64, 64), 128, dtype=np.uint8)
    regions = np.tile([0., 31.5, 31.5, 32., 32.], (30, 3, 1)).astype(np.float32)
    regions[0, :, 0] = -1
    metadata = np.zeros((30, 5), dtype=np.float32)
    metadata[1:, 0] = 1
    result = descriptor(volume, regions, metadata)
    assert result.shape == (25, FEATURE_COUNT)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:, 15:20], np.zeros((25, 5)))
    assert not result[0, 20:].any()
    assert result[1, 20] > 0 and result[1, 20 + 149] == result[1, 20]
