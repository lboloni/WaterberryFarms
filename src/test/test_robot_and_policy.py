import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from policy import FollowPathPolicy, RandomWaypointPolicy
from robot import Robot


class ObservationPolicy(FollowPathPolicy):
    def __init__(self):
        super().__init__(1, [[0, 0]], repeat=False)
        self.observations = []

    def add_observation(self, observation):
        self.observations.append(observation)


class TestRobotAndPolicy(unittest.TestCase):
    def test_pending_and_every_step_actions(self):
        robot = Robot("robot", 0, 0, 0)
        robot.add_action("East")
        robot.proceed(1)
        self.assertEqual((robot.x, robot.y), (1, 0))
        self.assertEqual(robot.pending_actions, [])
        self.assertEqual(list(robot.location_history)[0], [1, 0, 0])

    def test_location_velocity_and_acceleration_actions(self):
        robot = Robot("robot", 0, 0, 0)
        robot.enact_action("loc [2, 3, 4]")
        robot.enact_action("vel [1, -1, 2]")
        robot.enact_action("acc [2, 4, -2]", delta_t=0.5)
        self.assertEqual((robot.x, robot.y, robot.altitude), (2, 3, 4))
        self.assertEqual((robot.vel_x, robot.vel_y, robot.vel_altitude), (2, 1, 1))

    def test_follow_path_completion(self):
        robot = Robot("robot", 0, 0, 0)
        policy = FollowPathPolicy(1, [[0, 0], [2, 0]], repeat=False)
        robot.assign_policy(policy)
        for _ in range(3):
            robot.enact_policy(1)
            robot.proceed(1)
        self.assertEqual((robot.x, robot.y), (2, 0))
        self.assertEqual(policy.currentwaypoint, -1)

    def test_random_waypoint_is_seeded(self):
        robots = [Robot("robot-1", 0, 0, 0), Robot("robot-2", 0, 0, 0)]
        for robot in robots:
            robot.assign_policy(RandomWaypointPolicy(1, [0, 0], [10, 10], seed=7))
            robot.enact_policy(1)
        self.assertEqual(robots[0].policy.nextwaypoint, robots[1].policy.nextwaypoint)
        self.assertEqual(robots[0].pending_actions, robots[1].pending_actions)

    def test_observation_is_forwarded_to_policy(self):
        robot = Robot("robot", 0, 0, 0)
        policy = ObservationPolicy()
        robot.assign_policy(policy)
        robot.add_observation({"value": 1})
        self.assertEqual(policy.observations, [{"value": 1}])


if __name__ == "__main__":
    unittest.main()
