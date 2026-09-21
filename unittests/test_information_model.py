import pathlib
import sys
import unittest

import numpy as np
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from information_model import DiskEstimateScalarFieldIM, GaussianProcessScalarFieldIM, PointEstimateScalarFieldIM, StoredObservationIM


def observation(x, y, value):
    return {
        StoredObservationIM.X: x,
        StoredObservationIM.Y: y,
        StoredObservationIM.VALUE: value,
        StoredObservationIM.TIME: 0,
    }


class TestInformationModels(unittest.TestCase):
    def test_point_estimator_with_prior(self):
        estimator = PointEstimateScalarFieldIM(3, 3, default_value=0.25)
        prior_value = np.full((3, 3), 0.25)
        prior_uncertainty = np.ones((3, 3))
        value, uncertainty = estimator.estimate(
            [observation(1, 2, 0.8)], prior_value, prior_uncertainty)
        self.assertEqual(value[1, 2], 0.8)
        self.assertEqual(uncertainty[1, 2], 0.0)
        self.assertEqual(value[0, 0], 0.25)

    def test_point_estimator_uses_default_without_prior(self):
        estimator = PointEstimateScalarFieldIM(3, 3, default_value=0.25)
        value, uncertainty = estimator.estimate([], None, None)
        np.testing.assert_array_equal(value, np.full((3, 3), 0.25))
        np.testing.assert_array_equal(uncertainty, np.ones((3, 3)))

    def test_disk_is_centered_on_observation(self):
        estimator = DiskEstimateScalarFieldIM(5, 5, disk_radius=1)
        value, uncertainty = estimator.estimate(
            [observation(2, 2, 0.7)], None, None)
        expected = {(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)}
        actual = set(zip(*np.where(uncertainty == 0)))
        self.assertEqual(actual, expected)
        for location in expected:
            self.assertEqual(value[location], 0.7)

    def test_disk_clips_at_boundary_and_latest_value_wins(self):
        estimator = DiskEstimateScalarFieldIM(4, 4, disk_radius=1)
        value, uncertainty = estimator.estimate(
            [observation(0, 0, 0.2), observation(1, 0, 0.9)], None, None)
        self.assertEqual(value[0, 0], 0.9)
        self.assertEqual(uncertainty[0, 0], 0.0)
        self.assertEqual(uncertainty[3, 3], 1.0)

    def test_adaptive_disk_reduces_uncertainty(self):
        estimator = DiskEstimateScalarFieldIM(5, 5, disk_radius=None)
        voi = estimator.estimate_voi(observation(2, 2, 1.0))
        self.assertGreater(voi, 0)

    def test_gaussian_process_empty_and_observed_models(self):
        kernel = RBF([1.0, 1.0], length_scale_bounds="fixed") + WhiteKernel(
            1e-5, noise_level_bounds="fixed")
        estimator = GaussianProcessScalarFieldIM(3, 3, gp_kernel=kernel, default_value=0.5)
        value, uncertainty = estimator.estimate([], None, None)
        np.testing.assert_array_equal(value, np.full((3, 3), 0.5))
        np.testing.assert_array_equal(uncertainty, np.ones((3, 3)))

        value, uncertainty = estimator.estimate(
            [observation(0, 0, 0.0), observation(2, 2, 1.0)], None, None)
        self.assertEqual(value.shape, (3, 3))
        self.assertEqual(uncertainty.shape, (3, 3))
        self.assertLess(value[0, 0], value[2, 2])
        self.assertGreaterEqual(np.min(uncertainty), 0.0)


if __name__ == "__main__":
    unittest.main()
