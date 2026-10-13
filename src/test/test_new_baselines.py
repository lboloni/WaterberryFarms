"""The reimplemented baselines of the comparison study (paper-MRMR PLAN-ComparisonStudy.md): the adapted UCB
auction (papers/y2027_marex/ucb_auction.py, after Jia et al.) and the region-based task allocation
(papers/y2027_marex/region_task_allocation.py, after Edmonds & Yi)"""

import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium
from papers.y2027_marex.region_task_allocation import RegionTaskAllocationPolicy, estimate_kernel, partition
from papers.y2027_marex.team_knowledge import TeamKnowledge
from papers.y2027_marex.ucb_auction import UCBAuctionGP, auction, interest_points
from robot import Robot

GEOMETRY = {"width": 40, "height": 40, "velocity": 1}
WEIGHTS = (10.0, 2.0, 1.0, 5.0)


class TestUCBAuction(unittest.TestCase):
    def test_interest_points_are_unobserved_separated_and_in_ucb_order(self):
        ucb = np.zeros((40, 40))
        ucb[10, 10], ucb[11, 10], ucb[30, 30], ucb[5, 35] = 5, 4, 3, 2
        available = np.ones((40, 40), dtype=bool)
        available[30, 30] = False  # observed
        points = interest_points(ucb, available, 3, 7.5, np.random.default_rng(0))
        self.assertEqual(points[:2], [(10, 10), (5, 35)])  # (11, 10) is too close, (30, 30) observed
        self.assertTrue(all(math.dist(a, b) >= 7.5 for i, a in enumerate(points) for b in points[i + 1:]))

    def test_the_auction_gives_each_point_to_its_cheapest_idle_bidder(self):
        points = [(10, 10), (30, 30)]
        positions = {"a": (9, 9), "b": (29, 29), "c": (0, 0)}
        result = auction(points, lambda p: (1.0, 1.0), {"a", "b", "c"}, positions, 1000, [], 7.5, WEIGHTS)
        self.assertEqual(result, {"a": (10, 10), "b": (30, 30)})

    def test_a_robot_wins_at_most_one_point_per_round(self):
        result = auction([(10, 10), (30, 30)], lambda p: (1.0, 1.0), {"a"}, {"a": (9, 9)}, 1000, [], 7.5, WEIGHTS)
        self.assertEqual(result, {"a": (10, 10)})

    def test_points_beyond_the_budget_or_near_an_active_assignment_are_not_auctioned(self):
        self.assertEqual(auction([(30, 30)], lambda p: (1.0, 1.0), {"a"}, {"a": (0, 0)}, 20, [], 7.5, WEIGHTS), {})
        self.assertEqual(auction([(30, 30)], lambda p: (1.0, 1.0), {"a"}, {"a": (0, 0)}, 1000, [(32, 32)], 7.5,
                                 WEIGHTS), {})

    def test_the_gp_raises_the_mean_at_detections_and_lowers_the_uncertainty(self):
        knowledge = TeamKnowledge(40, 40, footprint=0)
        for x in range(15, 25):
            knowledge.add(x, 20, x in (19, 20, 21))
        gp = UCBAuctionGP(40, 40, lengthscale=2.5, inducing=8, healthy_value=-0.1)
        gp.refit(knowledge, optimize=True)
        self.assertGreater(gp.mu[20, 20], gp.mu[15, 20])
        self.assertLess(gp.sigma[20, 20], gp.sigma[5, 5])


def make_rbta(name, kernel, start=(0, 0), parameters=None):
    exp_policy = {"budget": 1000, "gamma-p": 0.03, "redundancy": "observed", **(parameters or {})}
    policy = RegionTaskAllocationPolicy(exp_policy, GEOMETRY, name, kernel)
    robot = Robot(name, *start, 0)
    robot.assign_policy(policy)
    return robot, policy


class TestRegionTaskAllocation(unittest.TestCase):
    def test_the_kernel_is_symmetric_maximal_at_the_origin_and_wider_for_clusters(self):
        clustered = estimate_kernel(60, 60, "clustered", 0.05, list(range(1001, 1011)))
        unclustered = estimate_kernel(60, 60, "unclustered", 0.05, list(range(1001, 1011)))
        for kernel in (clustered, unclustered):
            np.testing.assert_allclose(kernel, kernel[::-1, ::-1])
            r = kernel.shape[0] // 2
            self.assertEqual(kernel[r, r], kernel.max())
        self.assertGreater(clustered.shape[0], unclustered.shape[0])

    def test_the_partition_gives_every_cell_to_the_nearest_robot(self):
        names, owner = partition({"a": (5, 5), "b": (35, 35)}, {"a": 0, "b": 0}, 40, 40)
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(owner[0, 0], 0)
        self.assertEqual(owner[39, 39], 1)
        # a long remaining task time shrinks the region
        _, owner = partition({"a": (5, 5), "b": (35, 35)}, {"a": 0, "b": 100}, 40, 40)
        self.assertTrue((owner == 0).all())

    def test_with_a_flat_kernel_the_nearest_unobserved_cell_is_chosen(self):
        robot, policy = make_rbta("r-1", np.ones((1, 1)), start=(10, 10))
        policy.positions = {"r-1": (10, 10)}
        policy.used = 2
        target = policy.choose_task()
        self.assertEqual(math.dist(target, (10, 10)), 1.0)

    def test_robots_at_the_same_place_take_different_tasks(self):
        """The sequential allocation of the coordinator: free robots with the same knowledge do not follow
        each other"""
        kernel = estimate_kernel(40, 40, "clustered", 0.05, [1001, 1002])
        targets = []
        for name in ("r-1", "r-2", "r-3"):
            robot, policy = make_rbta(name, kernel)
            policy.positions = {"r-1": (0, 0), "r-2": (0, 0), "r-3": (0, 0)}
            policy.used = 2
            targets.append(policy.choose_task())
        self.assertEqual(len(set(targets)), 3)

    def test_the_task_is_announced_to_the_teammates(self):
        medium = PerfectCommunicationMedium(None)
        robots = [make_rbta(name, np.ones((1, 1)))[0] for name in ("r-1", "r-2")]
        for robot in robots:
            medium.add_robot(robot)
        medium.timestep, medium.round = 0, 0
        robots[0].policy.positions = {"r-1": (0, 0)}
        robots[0].policy.used = 2
        robots[0].policy.choose_task()
        robots[0].policy.act_send(0)
        robots[1].policy.act_receive(0, medium.receive(robots[1]))
        self.assertIn("r-1", robots[1].policy.tasks)


if __name__ == "__main__":
    unittest.main()
