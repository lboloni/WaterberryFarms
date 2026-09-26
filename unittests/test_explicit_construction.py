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
        self.assertFalse(hasattr(wbf_helper, "create_policy"))
        self.assertFalse(hasattr(wbf_helper, "create_estimator"))
        self.assertFalse(hasattr(wbf_helper, "create_score"))


if __name__ == "__main__":
    unittest.main()
