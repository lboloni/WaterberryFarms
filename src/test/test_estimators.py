import pathlib
import sys
import unittest
import warnings

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from information_model import OccupancyGridIM
from wbf_helper import create_field_estimator

DEFAULTS = yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[2] / "data" / "expruns" / "estimator" / "_defaults_estimator.yaml"))
TYPES = ["point", "disk", "gaussian-process", "gp-local", "gp-indicator", "nearest", "idw", "rbf",
         "occupancy", "mrf", "epidemic-pf", "cnn"]
# the GP regressions, whose posterior mean is not clipped to [0, 1]
UNCLIPPED = ["gaussian-process", "gp-local"]
# the estimators that keep the observed value at the observed cells
EXACT = ["point", "nearest", "idw", "rbf", "gp-indicator", "occupancy", "mrf", "epidemic-pf", "cnn"]


def observation(x, y, value):
    return {"x": x, "y": y, "value": value, "time": 0}


OBSERVATIONS = [observation(1, 1, 1.0), observation(3, 7, 0.5), observation(4, 7, 0.0),
                observation(8, 2, 1.0), observation(6, 6, 0.5), observation(2, 4, 1.0)]


def estimator(estimator_type, **parameters):
    return create_field_estimator(estimator_type, DEFAULTS | {"gp-restarts": 0, "pf-particles": 20} | parameters, 10, 10, 1.0)


class TestEstimatorContract(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore")

    def test_every_estimator_satisfies_the_contract(self):
        for estimator_type in TYPES:
            im = estimator(estimator_type)
            for obs in OBSERVATIONS:
                im.add_observation(obs)
            im.proceed(1)
            self.assertIn(im.UNCERTAINTY, ["coverage", "std", "probability", "distance", "none"], estimator_type)
            self.assertEqual(im.value.shape, (10, 10), estimator_type)
            self.assertEqual(im.uncertainty.shape, (10, 10), estimator_type)
            if estimator_type not in UNCLIPPED:
                self.assertTrue(np.all((im.value >= 0) & (im.value <= 1)), estimator_type)
            confidence = im.confidence()
            self.assertTrue(np.all((confidence >= 0) & (confidence <= 1)), estimator_type)
            if im.UNCERTAINTY == "probability":
                self.assertTrue(np.all((im.probability >= 0) & (im.probability <= 1)), estimator_type)
            if estimator_type in EXACT:
                for obs in OBSERVATIONS:
                    self.assertAlmostEqual(im.value[obs["x"], obs["y"]], obs["value"], msg=estimator_type)

    def test_without_observations_the_estimate_is_the_default(self):
        for estimator_type in ["point", "nearest", "idw", "rbf", "gaussian-process", "gp-local"]:
            im = estimator(estimator_type)
            im.proceed(1)
            np.testing.assert_array_equal(im.value, np.ones((10, 10)), estimator_type)
            self.assertTrue(np.all(im.confidence() == 0), estimator_type)

    def test_occupancy_evidence_spreads_to_the_neighbors(self):
        im = OccupancyGridIM(10, 10, footprint=2.0, prior=0.1)
        im.add_observation(observation(5, 5, 0.5))
        im.add_observation(observation(1, 1, 1.0))
        im.proceed(1)
        self.assertGreater(im.probability[5, 6], 0.1)   # next to the diseased cell
        self.assertLess(im.probability[1, 2], 0.1)      # next to the healthy cell
        self.assertAlmostEqual(im.probability[9, 0], 0.1, places=3)  # far from both

    def test_gp_confidence_is_normalized_by_the_prior(self):
        im = estimator("gaussian-process", **{"gp-normalize-y": True})
        for obs in OBSERVATIONS:
            im.add_observation(obs)
        im.proceed(1)
        self.assertGreater(im.prior_std, 0)
        self.assertGreater(im.confidence()[3, 7], im.confidence()[9, 9])


if __name__ == "__main__":
    unittest.main()
