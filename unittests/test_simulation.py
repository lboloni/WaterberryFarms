import gzip
import pathlib
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium
from policy import FollowPathPolicy
from robot import Robot
from wbf_simulate import save_simulation_results, simulate_1day, simulate_1day_multirobot


class RecordingEstimator:
    def __init__(self):
        self.observations = []
        self.proceed_calls = []

    def add_observation(self, observation):
        self.observations.append(observation)

    def proceed(self, delta_t):
        self.proceed_calls.append(delta_t)


class ObservationEnvironment:
    def __init__(self):
        self.time = 0

    def get_observation(self, position):
        return {"x": position[0], "y": position[1], "time": position[2]}


class ObservationCountScore:
    def score(self, environment, estimator):
        return len(estimator.observations)


def robot_with_path(name):
    robot = Robot(name, 0, 0, 0)
    robot.assign_policy(FollowPathPolicy(1, [[0, 0], [3, 0]], repeat=False))
    return robot


def base_results():
    return {
        "estimator-CODE": RecordingEstimator(),
        "score-code": ObservationCountScore(),
        "wbfe": ObservationEnvironment(),
        "timesteps-per-day": 4,
        "im_resolution": 3,
    }


class TestSimulation(unittest.TestCase):
    def test_single_robot_timestep_order_and_final_estimate(self):
        results = base_results()
        results["robot"] = robot_with_path("robot")
        simulate_1day(results)
        self.assertEqual(results["positions"], [[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3]])
        self.assertEqual(results["observations"], results["estimator-CODE"].observations)
        self.assertEqual(results["estimator-CODE"].proceed_calls, [3, 1])
        self.assertEqual(results["scores"], [3, 3, 3, 4])

    def test_repeated_runs_are_identical(self):
        runs = []
        for _ in range(2):
            results = base_results()
            results["robot"] = robot_with_path("robot")
            simulate_1day(results)
            runs.append((results["positions"], results["observations"], results["scores"]))
        self.assertEqual(runs[0], runs[1])

    def test_single_and_one_robot_multi_results_agree(self):
        single = base_results()
        single["robot"] = robot_with_path("robot")
        simulate_1day(single)

        multi = base_results()
        multi["robots"] = [robot_with_path("robot")]
        multi["communication"] = PerfectCommunicationMedium(multi["wbfe"])
        multi["communication"].add_robot(multi["robots"][0])
        multi["communication-rounds"] = 0
        simulate_1day_multirobot(multi)

        self.assertEqual(single["positions"], [positions[0] for positions in multi["positions"]])
        self.assertEqual(single["observations"], [observations[0] for observations in multi["observations"]])
        self.assertEqual(single["scores"], multi["scores"])

    def test_saved_results_exclude_runtime_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.pickle"
            save_simulation_results(path, {
                "value": 4,
                "score-code": lambda: None,
                "estimator-CODE": lambda: None,
            })
            with gzip.open(path, "rb") as handle:
                saved = pickle.load(handle)
            self.assertEqual(saved, {"value": 4})


if __name__ == "__main__":
    unittest.main()
