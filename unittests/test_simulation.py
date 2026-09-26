import gzip
import pathlib
import pickle
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import wbf_simulate
from communication import Message, PerfectCommunicationMedium
from policy import FollowPathPolicy
from robot import Robot
from water_berry_farm import (
    MiniberryFarm,
    WaterberryFarmEnvironment,
    WBF_IM_DiskEstimator,
    WBF_Score_WeightedAsymmetric,
)
from wbf_simulate import save_simulation_results, simulate_1day
from wbf_figures import show_detections, show_robot_path


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


class ObservationCountEvaluator:
    def score(self, environment, estimator):
        return len(estimator.observations)


def robot_with_path(name, y=0):
    robot = Robot(name, 0, y, 0)
    robot.assign_policy(FollowPathPolicy(1, [[0, y], [3, y]], repeat=False))
    return robot


class TestSimulation(unittest.TestCase):
    def test_single_robot_uses_canonical_nested_results(self):
        estimator = RecordingEstimator()
        results = simulate_1day(
            environment=ObservationEnvironment(),
            robots=[robot_with_path("robot")],
            estimator=estimator,
            evaluator=ObservationCountEvaluator(),
            timesteps=4,
            estimator_interval=3,
        )
        self.assertEqual(results["positions"], [
            [[0, 0, 0]], [[1, 0, 1]], [[2, 0, 2]], [[3, 0, 3]],
        ])
        self.assertEqual(
            [observations[0] for observations in results["observations"]],
            estimator.observations,
        )
        self.assertEqual(estimator.proceed_calls, [3, 1])
        self.assertEqual(results["score-events"], [
            {"timestep": 2, "score": 3},
            {"timestep": 3, "score": 4},
        ])
        self.assertIs(results["scores"], results["score-events"])

    def test_repeated_runs_are_identical(self):
        runs = []
        for _ in range(2):
            results = simulate_1day(
                environment=ObservationEnvironment(),
                robots=[robot_with_path("robot")],
                estimator=RecordingEstimator(),
                evaluator=ObservationCountEvaluator(),
                timesteps=4,
                estimator_interval=3,
            )
            runs.append((results["positions"], results["observations"],
                         results["score-events"]))
        self.assertEqual(runs[0], runs[1])

    def test_robot_configuration_order_does_not_change_results(self):
        runs = []
        for names in [("beta", "alpha"), ("alpha", "beta")]:
            robots = {
                "alpha": robot_with_path("alpha", 0),
                "beta": robot_with_path("beta", 1),
            }
            results = simulate_1day(
                environment=ObservationEnvironment(),
                robots=[robots[name] for name in names],
                estimator=RecordingEstimator(),
                evaluator=ObservationCountEvaluator(),
                timesteps=4,
                estimator_interval=3,
            )
            runs.append((results["robot-names"], results["positions"],
                         results["observations"], results["score-events"]))
        self.assertEqual(runs[0], runs[1])

    def test_hooks_receive_explicit_runtime_objects(self):
        environment = ObservationEnvironment()
        robots = [robot_with_path("robot")]
        estimator = RecordingEstimator()
        evaluator = ObservationCountEvaluator()
        timestep_calls = []
        day_calls = []

        def after_timestep(results, hook_environment, hook_robots,
                           hook_estimator, hook_evaluator):
            self.assertIs(hook_environment, environment)
            self.assertEqual(hook_robots, results["robots"])
            self.assertIs(hook_estimator, estimator)
            self.assertIs(hook_evaluator, evaluator)
            timestep_calls.append(len(results["observations"]))

        def after_day(results, hook_environment, hook_robots,
                      hook_estimator, hook_evaluator):
            self.assertIs(hook_environment, environment)
            self.assertEqual(hook_robots, results["robots"])
            self.assertIs(hook_estimator, estimator)
            self.assertIs(hook_evaluator, evaluator)
            day_calls.append(len(results["observations"]))

        simulate_1day(
            environment=environment,
            robots=robots,
            estimator=estimator,
            evaluator=evaluator,
            timesteps=4,
            estimator_interval=3,
            after_timestep=after_timestep,
            after_day=after_day,
        )
        self.assertEqual(timestep_calls, [1, 2, 3, 4])
        self.assertEqual(day_calls, [4])

    def test_duplicate_names_and_missing_policies_fail(self):
        with self.assertRaisesRegex(Exception, "unique"):
            simulate_1day(
                environment=ObservationEnvironment(),
                robots=[robot_with_path("same"), robot_with_path("same")],
                estimator=RecordingEstimator(),
                evaluator=ObservationCountEvaluator(),
                timesteps=1,
                estimator_interval=1,
            )

        with self.assertRaisesRegex(Exception, "no policy"):
            simulate_1day(
                environment=ObservationEnvironment(),
                robots=[Robot("unassigned", 0, 0, 0)],
                estimator=RecordingEstimator(),
                evaluator=ObservationCountEvaluator(),
                timesteps=1,
                estimator_interval=1,
            )

    def test_environment_must_remain_static_during_the_day(self):
        class AdvancingEnvironment(ObservationEnvironment):
            def get_observation(self, position):
                self.time += 1
                return super().get_observation(position)

        with self.assertRaisesRegex(Exception, "Environment time changed"):
            simulate_1day(
                environment=AdvancingEnvironment(),
                robots=[robot_with_path("robot")],
                estimator=RecordingEstimator(),
                evaluator=ObservationCountEvaluator(),
                timesteps=1,
                estimator_interval=1,
            )

        environment = ObservationEnvironment()

        def advance_environment(results, hook_environment, robots, estimator,
                                evaluator):
            hook_environment.time += 1

        with self.assertRaisesRegex(Exception, "Environment time changed"):
            simulate_1day(
                environment=environment,
                robots=[robot_with_path("robot")],
                estimator=RecordingEstimator(),
                evaluator=ObservationCountEvaluator(),
                timesteps=1,
                estimator_interval=1,
                after_timestep=advance_environment,
            )

    def test_miniberry_adaptive_disk_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            farm = MiniberryFarm(scale=1)
            farm.create_type_map()
            environment = WaterberryFarmEnvironment(
                farm, use_saved=False, seed=10, savedir=directory)
            results = simulate_1day(
                environment=environment,
                robots=[robot_with_path("robot")],
                estimator=WBF_IM_DiskEstimator(11, 11),
                evaluator=WBF_Score_WeightedAsymmetric(),
                timesteps=2,
                estimator_interval=2,
            )
            self.assertEqual(len(results["observations"]), 2)
            self.assertEqual(results["score-events"][0]["timestep"], 1)

            figure, axes = plt.subplots()
            show_robot_path(results, axes, draw_robot=False)
            show_detections(results, axes)
            plt.close(figure)

    def test_persistence_saves_the_caller_selected_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.pickle"
            save_simulation_results(path, {"value": 4})
            with gzip.open(path, "rb") as handle:
                saved = pickle.load(handle)
            self.assertEqual(saved, {"value": 4})

    def test_legacy_entry_points_are_removed(self):
        self.assertFalse(hasattr(wbf_simulate, "run_1robot1day"))
        self.assertFalse(hasattr(wbf_simulate, "run_nrobot1day"))
        self.assertFalse(hasattr(wbf_simulate, "simulate_1day_multirobot"))
        self.assertFalse(hasattr(wbf_simulate, "simulate_timestep_1robot"))
        self.assertFalse(hasattr(wbf_simulate, "simulate_timestep_multirobot"))


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


class TraceEvaluator:
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


class TraceCommunicationPolicy:
    """Communication policy intentionally unrelated to framework base classes."""

    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def act(self, delta_t):
        self.trace.append(("policy", self.name))

    def add_observation(self, observation):
        pass

    def act_send(self, round_number):
        self.trace.append(("send", round_number, self.name))
        self.robot.com.send(self.robot, None, Message(round_number))

    def act_receive(self, round_number, messages):
        self.trace.append((
            "receive", round_number, self.name,
            [message.sender_name for message in messages],
        ))


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


def trace_components(communicating=False):
    trace = []
    policy_type = TraceCommunicationPolicy if communicating else TracePolicy
    robots = [
        TraceRobot("beta", 10, policy_type("beta", trace), trace),
        TraceRobot("alpha", 0, policy_type("alpha", trace), trace),
    ]
    environment = TraceEnvironment(trace)
    estimator = TraceEstimator(trace)
    evaluator = TraceEvaluator(trace)
    communication = PerfectCommunicationMedium(environment)
    for robot in robots:
        communication.add_robot(robot)
    return trace, robots, environment, estimator, evaluator, communication


class TestCanonicalLifecycle(unittest.TestCase):
    def test_exact_phase_order(self):
        trace, robots, environment, estimator, evaluator, communication = \
            trace_components()

        def after_timestep(results, environment, robots, estimator, evaluator):
            trace.append(("timestep-hook", results["simulation-timestep"]))

        def after_day(results, environment, robots, estimator, evaluator):
            trace.append(("day-hook", results["simulation-timestep"]))

        simulate_1day(
            environment=environment,
            robots=robots,
            estimator=estimator,
            evaluator=evaluator,
            timesteps=1,
            estimator_interval=1,
            after_timestep=after_timestep,
            after_day=after_day,
        )
        self.assertEqual(trace, [
            ("policy", "alpha", 0), ("policy", "beta", 0),
            ("execute", "alpha"), ("execute", "beta"),
            ("observe", "alpha"), ("observe", "beta"),
            ("estimator-add", "alpha"), ("estimator-add", "beta"),
            ("policy-observation", "alpha"),
            ("policy-observation", "beta"),
            ("estimate", 1), ("score", 2),
            ("timestep-hook", 1), ("day-hook", 1),
        ])

    def test_policy_cannot_use_current_observation(self):
        trace, robots, environment, estimator, evaluator, communication = \
            trace_components()
        simulate_1day(
            environment=environment,
            robots=robots,
            estimator=estimator,
            evaluator=evaluator,
            timesteps=2,
            estimator_interval=2,
        )
        policy_calls = [event for event in trace if event[0] == "policy"]
        self.assertEqual(policy_calls, [
            ("policy", "alpha", 0), ("policy", "beta", 0),
            ("policy", "alpha", 1), ("policy", "beta", 1),
        ])

    def test_communication_uses_plain_external_policy(self):
        trace, robots, environment, estimator, evaluator, communication = \
            trace_components(communicating=True)
        simulate_1day(
            environment=environment,
            robots=robots,
            estimator=estimator,
            evaluator=evaluator,
            timesteps=1,
            estimator_interval=1,
            communication=communication,
            communication_rounds=2,
        )
        communication_events = [
            event for event in trace if event[0] in ("send", "receive")
        ]
        self.assertEqual(communication_events, [
            ("send", 0, "alpha"), ("send", 0, "beta"),
            ("receive", 0, "alpha", ["beta"]),
            ("receive", 0, "beta", ["alpha"]),
            ("send", 1, "alpha"), ("send", 1, "beta"),
            ("receive", 1, "alpha", ["beta"]),
            ("receive", 1, "beta", ["alpha"]),
        ])


if __name__ == "__main__":
    unittest.main()
