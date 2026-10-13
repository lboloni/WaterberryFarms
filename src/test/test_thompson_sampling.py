"""The common infrastructure of the reimplemented baselines (papers/y2027_marex/team_knowledge.py) and the
decentralized Thompson sampling baseline (papers/y2027_marex/thompson_sampling.py, after GUTS)"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium
from papers.y2027_marex.team_knowledge import TeamKnowledge, path_cells
from papers.y2027_marex.thompson_sampling import ThompsonSamplingPolicy
from policy import FollowPathPolicy
from robot import Robot

GEOMETRY = {"width": 100, "height": 100, "velocity": 1}
PARAMETERS = {"budget": 1000, "seed": 1, "candidates": 256, "prior": 0.05, "p-hit": 0.99,
              "p-false-alarm": 0.01, "footprint": 2.0, "policy-name": "ts"}


def observation(x, y, diseased):
    return {"x": x, "y": y, "time": 0, "TYLCV": {"x": x, "y": y, "time": 0, "value": 0.5 if diseased else 1.0}}


class TestPathCells(unittest.TestCase):
    def test_the_cells_are_those_the_simulator_records(self):
        """A robot that follows a straight path observes exactly the cells of path_cells"""
        for start, end in [((0, 0), (7, 3)), ((10, 10), (10, 2)), ((5.0, 5.0), (6, 6)), ((0, 0), (25, 40))]:
            robot = Robot("r", *start, 0)
            robot.assign_policy(FollowPathPolicy(vel=1, waypoints=[list(end)]))
            visited = []
            while not (robot.x == end[0] and robot.y == end[1]):
                robot.enact_policy(1)
                robot.proceed(1)
                visited.append((int(robot.x), int(robot.y)))
            self.assertEqual([tuple(c) for c in path_cells(start, end)], visited, (start, end))

    def test_a_path_to_the_start_observes_nothing(self):
        self.assertEqual(len(path_cells((3, 4), (3, 4))), 0)


class TestTeamKnowledge(unittest.TestCase):
    def test_a_detection_raises_its_neighbors_with_a_footprint_only(self):
        spatial, independent = TeamKnowledge(20, 20, footprint=2.0), TeamKnowledge(20, 20, footprint=0)
        for knowledge in (spatial, independent):
            knowledge.add(10, 10, True)
        self.assertEqual(spatial.probability()[10, 10], 1.0)
        self.assertGreater(spatial.probability()[11, 10], 0.5)
        self.assertAlmostEqual(independent.probability()[11, 10], 0.05)
        # a healthy observation lowers the probability around it
        spatial.add(2, 2, False)
        self.assertLess(spatial.probability()[3, 2], 0.05)

    def test_a_cell_counts_once(self):
        knowledge = TeamKnowledge(20, 20)
        self.assertTrue(knowledge.add(5, 5, True))
        before = knowledge.logodds.copy()
        self.assertFalse(knowledge.add(5, 5, True))
        np.testing.assert_array_equal(knowledge.logodds, before)


def make_robot(parameters=None, start=(0, 0), name="ts-1"):
    policy = ThompsonSamplingPolicy({**PARAMETERS, **(parameters or {})}, GEOMETRY)
    robot = Robot(name, *start, 0)
    robot.assign_policy(policy)
    return robot, policy


class TestThompsonSampling(unittest.TestCase):
    def test_after_a_detection_the_spatial_posterior_stays_near_it(self):
        """With a footprint, the waypoints after a detection go near it; with independent cells, they do not"""
        distances = {}
        for footprint in (2.0, 0.0):
            near = []
            for seed in range(20):
                robot, policy = make_robot({"footprint": footprint, "seed": seed}, start=(50, 50))
                policy.add_observation(observation(50, 50, True))
                waypoint = policy.choose_waypoint()
                near.append(np.hypot(waypoint[0] - 50, waypoint[1] - 50))
            distances[footprint] = np.median(near)
        # the spatial posterior goes to the neighbors of the detection; the independent one only to the
        # nearest cell its sample says is diseased, wherever it is
        self.assertLess(distances[2.0], 2.5)
        self.assertLess(distances[2.0], distances[0.0] / 2)

    def test_without_sampled_targets_the_nearest_candidate_wins(self):
        robot, policy = make_robot({"footprint": 0, "prior": 1e-9}, start=(50, 50))
        waypoint = policy.choose_waypoint()
        # no cell is sampled diseased: every candidate scores 0, and the tie goes to the nearest
        self.assertLessEqual(np.hypot(waypoint[0] - 50, waypoint[1] - 50), 5)

    def test_a_waypoint_never_exceeds_the_remaining_budget(self):
        robot, policy = make_robot({"budget": 30})
        policy.used = 20
        for _ in range(20):
            waypoint = policy.choose_waypoint()
            self.assertLessEqual(np.hypot(*waypoint), 10)
        policy.used = 30
        self.assertIsNone(policy.choose_waypoint())

    def test_the_decisions_are_reproducible_from_the_seed(self):
        first = [make_robot({"seed": 7})[1].choose_waypoint() for _ in range(2)]
        self.assertEqual(first[0], first[1])

    def test_the_observations_are_shared_with_the_teammates(self):
        medium = PerfectCommunicationMedium(None)
        robots = [make_robot(name=f"ts-{k}")[0] for k in (1, 2)]
        for robot in robots:
            medium.add_robot(robot)
        medium.timestep, medium.round = 0, 0
        robots[0].policy.add_observation(observation(3, 4, True))
        for robot in robots:
            robot.policy.act_send(0)
        robots[1].policy.act_receive(0, medium.receive(robots[1]))
        self.assertTrue(robots[1].policy.knowledge.observed[3, 4])
        self.assertTrue(robots[1].policy.knowledge.diseased[3, 4])
        self.assertEqual(medium.traffic[0]["type"], "observation")


if __name__ == "__main__":
    unittest.main()
