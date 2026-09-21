import gzip
import pathlib
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium
from communication import Message
from policy import AbstractCommunicateAndFollowPath, FollowPathPolicy
from robot import Robot
from wbf_simulate import save_simulation_results, simulate_1day, simulate_1day_multirobot
from water_berry_farm import MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_DiskEstimator, WBF_Score_WeightedAsymmetric


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


def robot_with_path(name, y=0):
    robot = Robot(name, 0, y, 0)
    robot.assign_policy(FollowPathPolicy(1, [[0, y], [3, y]], repeat=False))
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
        self.assertEqual(results["score-events"], [
            {"timestep": 2, "score": 3},
            {"timestep": 3, "score": 4},
        ])
        self.assertIs(results["scores"], results["score-events"])

    def test_repeated_runs_are_identical(self):
        runs = []
        for _ in range(2):
            results = base_results()
            results["robot"] = robot_with_path("robot")
            simulate_1day(results)
            runs.append((results["positions"], results["observations"], results["score-events"]))
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
        self.assertEqual(single["score-events"], multi["score-events"])

    def test_robot_configuration_order_does_not_change_results(self):
        runs = []
        for names in [("beta", "alpha"), ("alpha", "beta")]:
            results = base_results()
            robots = {
                "alpha": robot_with_path("alpha", 0),
                "beta": robot_with_path("beta", 1),
            }
            results["robots"] = [robots[name] for name in names]
            results["communication"] = PerfectCommunicationMedium(results["wbfe"])
            for robot in results["robots"]:
                results["communication"].add_robot(robot)
            results["communication-rounds"] = 0
            simulate_1day_multirobot(results)
            runs.append((results["robot-names"], results["positions"],
                         results["observations"], results["score-events"]))
        self.assertEqual(runs[0], runs[1])

    def test_timestep_and_day_hooks_are_distinct(self):
        results = base_results()
        results["robot"] = robot_with_path("robot")
        timestep_calls = []
        day_calls = []
        results["hook-after-timestep"] = lambda state: timestep_calls.append(len(state["observations"]))
        results["hook-after-day"] = lambda state: day_calls.append(len(state["observations"]))
        simulate_1day(results)
        self.assertEqual(timestep_calls, [1, 2, 3, 4])
        self.assertEqual(day_calls, [4])

    def test_duplicate_names_and_missing_policies_fail(self):
        results = base_results()
        results["robots"] = [robot_with_path("same"), robot_with_path("same")]
        results["communication-rounds"] = 0
        with self.assertRaisesRegex(Exception, "unique"):
            simulate_1day_multirobot(results)

        results = base_results()
        results["robots"] = [Robot("unassigned", 0, 0, 0)]
        results["communication-rounds"] = 0
        with self.assertRaisesRegex(Exception, "no policy"):
            simulate_1day_multirobot(results)

    def test_environment_must_remain_static_during_the_day(self):
        class AdvancingEnvironment(ObservationEnvironment):
            def get_observation(self, position):
                self.time += 1
                return super().get_observation(position)

        results = base_results()
        results["wbfe"] = AdvancingEnvironment()
        results["robot"] = robot_with_path("robot")
        with self.assertRaisesRegex(Exception, "Environment time changed"):
            simulate_1day(results)

        results = base_results()
        results["robot"] = robot_with_path("robot")
        results["hook-after-timestep"] = lambda state: setattr(state["wbfe"], "time", 1)
        with self.assertRaisesRegex(Exception, "Environment time changed"):
            simulate_1day(results)

    def test_miniberry_adaptive_disk_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            farm = MiniberryFarm(scale=1)
            farm.create_type_map()
            environment = WaterberryFarmEnvironment(
                farm, use_saved=False, seed=10, savedir=directory)
            results = {
                "estimator-CODE": WBF_IM_DiskEstimator(11, 11),
                "score-code": WBF_Score_WeightedAsymmetric(),
                "wbfe": environment,
                "timesteps-per-day": 2,
                "im_resolution": 2,
                "robot": robot_with_path("robot"),
            }
            simulate_1day(results)
            self.assertEqual(len(results["observations"]), 2)
            self.assertEqual(results["score-events"][0]["timestep"], 1)

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


class TraceEstimator:
    def __init__(self, trace):
        self.trace = trace
        self.observations = []

    def add_observation(self, observation):
        self.trace.append(("estimator-add", observation["robot"]))
        self.observations.append(observation)

    def proceed(self, delta_t):
        self.trace.append(("estimate", delta_t))


class TraceEnvironment:
    def __init__(self, trace):
        self.trace = trace
        self.time = 7

    def get_observation(self, position):
        name = "alpha" if position[0] < 10 else "beta"
        self.trace.append(("observe", name))
        return {"robot": name, "time": position[2]}


class TraceScore:
    def __init__(self, trace):
        self.trace = trace

    def score(self, environment, estimator):
        self.trace.append(("score", len(estimator.observations)))
        return len(estimator.observations)


class TracePolicy:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace
        self.observations = []

    def act(self, delta_t):
        self.trace.append(("policy", self.name, len(self.observations)))

    def add_observation(self, observation):
        self.trace.append(("policy-observation", self.name))
        self.observations.append(observation)


class TraceCommunicationPolicy(AbstractCommunicateAndFollowPath):
    def __init__(self, name, trace):
        super().__init__(1, [[0, 0]], repeat=False)
        self.name = name
        self.trace = trace

    def act(self, delta_t):
        self.trace.append(("policy", self.name))

    def act_send(self, round):
        self.trace.append(("send", round, self.name))
        self.robot.com.send(self.robot, None, Message(round))

    def act_receive(self, round, messages):
        self.trace.append(("receive", round, self.name,
                           [message.sender_name for message in messages]))


class TraceRobot:
    def __init__(self, name, x, policy, trace):
        self.name = name
        self.x = x
        self.y = 0
        self.policy = policy
        self.policy.robot = self
        self.trace = trace
        self.im = None
        self.com = None

    def enact_policy(self):
        self.policy.act(1)

    def proceed(self, delta_t):
        self.trace.append(("execute", self.name))
        self.x += 1

    def add_observation(self, observation):
        self.policy.add_observation(observation)


def trace_results(communicating=False, timesteps=1, im_resolution=1):
    trace = []
    policy_type = TraceCommunicationPolicy if communicating else TracePolicy
    robots = [
        TraceRobot("beta", 10, policy_type("beta", trace), trace),
        TraceRobot("alpha", 0, policy_type("alpha", trace), trace),
    ]
    environment = TraceEnvironment(trace)
    communication = PerfectCommunicationMedium(environment)
    for robot in robots:
        communication.add_robot(robot)
    results = {
        "robots": robots,
        "communication": communication,
        "communication-rounds": 2 if communicating else 0,
        "estimator-CODE": TraceEstimator(trace),
        "score-code": TraceScore(trace),
        "wbfe": environment,
        "timesteps-per-day": timesteps,
        "im_resolution": im_resolution,
    }
    return results, trace


class TestCanonicalLifecycle(unittest.TestCase):
    def test_exact_phase_order(self):
        results, trace = trace_results()
        results["hook-after-timestep"] = lambda state: trace.append(("timestep-hook", state["simulation-timestep"]))
        results["hook-after-day"] = lambda state: trace.append(("day-hook", state["simulation-timestep"]))
        simulate_1day_multirobot(results)
        self.assertEqual(trace, [
            ("policy", "alpha", 0), ("policy", "beta", 0),
            ("execute", "alpha"), ("execute", "beta"),
            ("observe", "alpha"), ("observe", "beta"),
            ("estimator-add", "alpha"), ("estimator-add", "beta"),
            ("policy-observation", "alpha"), ("policy-observation", "beta"),
            ("estimate", 1), ("score", 2),
            ("timestep-hook", 1), ("day-hook", 1),
        ])

    def test_policy_cannot_use_current_observation(self):
        results, trace = trace_results(timesteps=2, im_resolution=2)
        simulate_1day_multirobot(results)
        policy_calls = [event for event in trace if event[0] == "policy"]
        self.assertEqual(policy_calls, [
            ("policy", "alpha", 0), ("policy", "beta", 0),
            ("policy", "alpha", 1), ("policy", "beta", 1),
        ])

    def test_communication_rounds_have_send_receive_barriers(self):
        results, trace = trace_results(communicating=True)
        simulate_1day_multirobot(results)
        communication = [event for event in trace if event[0] in ("send", "receive")]
        self.assertEqual(communication, [
            ("send", 0, "alpha"), ("send", 0, "beta"),
            ("receive", 0, "alpha", ["beta"]),
            ("receive", 0, "beta", ["alpha"]),
            ("send", 1, "alpha"), ("send", 1, "beta"),
            ("receive", 1, "alpha", ["beta"]),
            ("receive", 1, "beta", ["alpha"]),
        ])


if __name__ == "__main__":
    unittest.main()
