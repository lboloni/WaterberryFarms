import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import wbf_helper
from papers.y2023_confidenceguided.confidence_guided_ipp_policy import (
    ConfidenceGuidedPathPlanning,
    generate_confidence_guided_ipp_policy,
)
from papers.y2023_glr.grid_limited_randomness import (
    generate_GLR_CA,
    generate_GLR_EOP,
    generate_GLR_SD,
)
from policy import FollowPathPolicy
from water_berry_farm import WBF_IM_DiskEstimator, WBF_IM_GaussianProcess
from wbf_helper import generate_fixed_budget_lawnmower


class TestExplicitConstruction(unittest.TestCase):
    def test_policy_generators_are_called_directly(self):
        environment = {"typename": "Miniberry-10"}

        lawnmower = generate_fixed_budget_lawnmower(
            {"budget": 40, "policy-name": "lawnmower"}, environment)
        glr_eop = generate_GLR_EOP(
            {"seed": 1, "h_cells": 2, "v_cells": 2}, environment)
        glr_sd = generate_GLR_SD(
            {"seed": 1, "h_cells": 2, "v_cells": 2}, environment)
        glr_ca = generate_GLR_CA(
            {"seed": 1, "h_cells": 2, "v_cells": 2}, environment)
        confidence_guided = generate_confidence_guided_ipp_policy(
            {"span": 2, "policy-name": "confidence-guided"}, environment)

        self.assertIsInstance(lawnmower, FollowPathPolicy)
        self.assertIsInstance(glr_eop, FollowPathPolicy)
        self.assertIsInstance(glr_sd, FollowPathPolicy)
        self.assertIsInstance(glr_ca, FollowPathPolicy)
        self.assertIsInstance(
            confidence_guided, ConfidenceGuidedPathPlanning)

    def test_central_component_dispatch_is_removed(self):
        """The estimator is the only component created through a factory"""
        self.assertFalse(hasattr(wbf_helper, "create_policy"))
        self.assertFalse(hasattr(wbf_helper, "create_score"))

    def test_estimator_factory(self):
        geometry = {"width": 10, "height": 10}
        defaults = {"default-tylcv": 1.0, "default-ccr": 1.0, "default-soil": 0.0}
        disk = wbf_helper.create_estimator(
            defaults | {"estimator-name": "D5", "estimator-type": "disk", "disk-radius": 5}, geometry)
        gp = wbf_helper.create_estimator(
            defaults | {"estimator-name": "GP", "estimator-type": "gaussian-process",
                        "gp-length-scale": 2.0, "gp-length-scale-bounds": [1, 10],
                        "gp-noise": 0.5, "gp-restarts": 0, "gp-normalize-y": True}, geometry)

        self.assertIsInstance(disk, WBF_IM_DiskEstimator)
        self.assertEqual(disk.name, "D5")
        self.assertEqual(disk.im_tylcv.disk_radius, 5)
        self.assertIsInstance(gp, WBF_IM_GaussianProcess)
        self.assertEqual(gp.im_ccr.n_restarts_optimizer, 0)
        self.assertTrue(gp.im_soil.normalize_y)
        with self.assertRaises(Exception):
            wbf_helper.create_estimator(defaults | {"estimator-type": "unknown"}, geometry)


if __name__ == "__main__":
    unittest.main()
