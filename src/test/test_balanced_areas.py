"""The balanced exploration areas of the multi-robot baselines (papers/y2027_marex/balanced_areas.py, 
DESIGN-BALANCED-EXPLORATION-AREAS.md)"""

import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from papers.y2027_marex.balanced_areas import (AreaConfidenceGuided, TransitThenExplorePolicy, allocate_balanced_strips,
    create_balanced_policy, lawnmower_passes, pass_cost)
from path_generators import get_path_length
from robot import Robot

AREA_200 = [10, 10, 190, 190]


def allocate(robots, area=AREA_200, budget=1000):
    return allocate_balanced_strips([(0, 0)] * robots, area, 1, budget)


class TestAllocation(unittest.TestCase):
    def test_the_configurations_of_the_paper(self):
        """The allocations of DESIGN-BALANCED-EXPLORATION-AREAS.md, Section 3.2"""
        for robots, area, passes, spacing in [(3, [5, 5, 95, 95], [10, 10, 10], 90 / 29),
                                              (3, AREA_200, [5, 4, 4], 15.0),
                                              (6, AREA_200, [5, 5, 4, 4, 4, 4], 180 / 25)]:
            allocation = allocate(robots, area)
            self.assertEqual([a["passes"] for a in allocation], passes)
            self.assertAlmostEqual(allocation[0]["spacing"], spacing)

    def test_every_robot_stays_within_its_budget(self):
        for robots in [1, 2, 3, 6, 10]:
            for a in allocate(robots):
                if a["passes"]:
                    self.assertLessEqual(pass_cost(a["travel"], a["passes"], 180, a["spacing"]), 1000 + 1e-6)

    def test_the_passes_are_uniform_and_cover_the_area(self):
        for robots in [2, 3, 6]:
            allocation = allocate(robots)
            ys = [y for a in allocation for y in a["ys"]]
            self.assertAlmostEqual(ys[0], 10)
            self.assertAlmostEqual(ys[-1], 190)
            self.assertTrue(np.allclose(np.diff(ys), allocation[0]["spacing"]))
            # the strips touch and cover the area
            strips = [a["area"] for a in allocation]
            self.assertAlmostEqual(strips[0][1], 10)
            self.assertAlmostEqual(strips[-1][3], 190)
            for lower, upper in zip(strips, strips[1:]):
                self.assertAlmostEqual(lower[3], upper[1])

    def test_farther_robots_get_no_more_passes(self):
        passes = [a["passes"] for a in allocate(6)]
        self.assertEqual(passes, sorted(passes, reverse=True))

    def test_a_too_small_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            allocate(1, budget=100)

    def test_the_lawnmower_costs_what_the_allocation_assumed(self):
        for a in allocate(3):
            path = lawnmower_passes(a, AREA_200)
            self.assertEqual(list(path[0]), a["entry"])
            self.assertAlmostEqual(get_path_length(path, [0, 0]), pass_cost(a["travel"], a["passes"], 180, a["spacing"]))


class EmptyEstimator:
    """The uncertainty of an estimator that has seen nothing, except one certain cell"""
    class Field:
        def __init__(self):
            self.uncertainty = np.ones((200, 200))
    def __init__(self):
        self.im_tylcv = self.Field()
    def proceed(self, delta_t):
        pass
    def add_observation(self, obs):
        pass


class TestPolicies(unittest.TestCase):
    def fly(self, policy, timesteps):
        robot = Robot("r", 0, 0, 0)
        robot.assign_policy(policy)
        trajectory = []
        for _ in range(timesteps):
            robot.enact_policy(1)
            robot.proceed(1)
            trajectory.append((robot.x, robot.y))
        return trajectory

    def test_the_robots_fly_to_their_entry_then_stay_in_their_area(self):
        allocation = allocate(3)
        far = allocation[-1]
        for run_policy in ["balanced-lawnmower", "balanced-random-waypoint", "balanced-confidence-guided"]:
            policy = create_balanced_policy(run_policy, {"policy-name": "p", "seed": 1, "span": 5}, far, AREA_200, 1,
                                            EmptyEstimator())
            self.assertIsInstance(policy, TransitThenExplorePolicy)
            trajectory = self.fly(policy, 1000)
            arrival = math.ceil(far["travel"])
            self.assertTrue(np.allclose(trajectory[arrival - 1], far["entry"]), run_policy)
            x_min, y_min, x_max, y_max = far["area"]
            for x, y in trajectory[arrival:]:
                self.assertTrue(x_min - 1e-6 <= x <= x_max + 1e-6 and y_min - 1 <= y <= y_max + 1, (run_policy, x, y))

    def test_the_confidence_guided_explorer_prefers_uncertain_cells(self):
        estimator = EmptyEstimator()
        estimator.im_tylcv.uncertainty[:, :] = 0.1
        estimator.im_tylcv.uncertainty[12, 13] = 1.0
        explorer = AreaConfidenceGuided(1, [10, 10], [20, 20], estimator, span=5, seed=0)
        explorer.assign(Robot("r", 10, 10, 0))
        self.assertEqual(explorer.choose_next_waypoint(explorer.generate_feasible_waypoints()), (12, 13))
        # far from its area, the explorer heads to the nearest cell of the area
        explorer.assign(Robot("r", 0, 0, 0))
        self.assertEqual(explorer.generate_feasible_waypoints(), [(10, 10)])


if __name__ == "__main__":
    unittest.main()
