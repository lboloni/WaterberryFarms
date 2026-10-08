import pathlib
import sys
import time
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from estimator_benchmark import evaluate, field_metrics, generate_observations
from information_model import PointEstimateScalarFieldIM
from water_berry_farm import MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_Composite


def environment():
    farm = MiniberryFarm(scale=1)
    farm.create_type_map()
    env = WaterberryFarmEnvironment(farm, use_saved=False, seed=10, savedir=None)
    env.proceed(6)
    return env


class TestSamplers(unittest.TestCase):
    def test_samplers_are_deterministic_and_stay_on_the_farm(self):
        env = environment()
        for sampler in [{"kind": "random", "count": 30}, {"kind": "random-waypoint", "timesteps": 30},
                        {"kind": "lawnmower", "timesteps": 60}]:
            first = generate_observations(sampler, env, "Miniberry-10", 3)
            second = generate_observations(sampler, env, "Miniberry-10", 3)
            self.assertEqual([(o["x"], o["y"]) for o in first], [(o["x"], o["y"]) for o in second], sampler)
            self.assertTrue(all(0 <= o["x"] < 10 and 0 <= o["y"] < 10 for o in first), sampler)
        cells = [(o["x"], o["y"]) for o in generate_observations({"kind": "random", "count": 30}, env, "Miniberry-10", 0)]
        self.assertEqual(len(set(cells)), 30)


class TestMetrics(unittest.TestCase):
    def test_disease_metrics_on_a_known_case(self):
        truth = SimpleNamespace(value=np.array([[1.0, 0.5], [0.0, 1.0]]))
        estimate = SimpleNamespace(value=np.array([[1.0, 0.75], [0.25, 1.0]]), uncertainty=np.zeros((2, 2)),
                                   UNCERTAINTY="probability", probability=np.array([[0.2, 0.6], [0.9, 0.4]]),
                                   confidence=lambda: np.ones((2, 2)))
        metrics = field_metrics(truth, estimate, np.ones((2, 2), dtype=bool), True)
        self.assertAlmostEqual(metrics["mae"], (0 + 0.25 + 0.25 + 0) / 4)
        self.assertAlmostEqual(metrics["brier"], (0.2**2 + 0.4**2 + 0.1**2 + 0.4**2) / 4)
        self.assertEqual(metrics["recall"], 1.0)        # both diseased cells predicted (p >= 0.5)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["native-probability"], 1.0)

    def test_std_interval_coverage(self):
        truth = SimpleNamespace(value=np.array([[0.0, 0.0, 0.0, 0.0]]))
        estimate = SimpleNamespace(value=np.array([[0.05, 0.15, 0.25, 0.5]]), uncertainty=np.full((1, 4), 0.1),
                                   UNCERTAINTY="std", probability=None, confidence=lambda: np.full((1, 4), 0.5))
        metrics = field_metrics(truth, estimate, np.ones((1, 4), dtype=bool), False)
        self.assertEqual(metrics["coverage-1sd"], 0.25)
        self.assertEqual(metrics["coverage-2sd"], 0.5)


class SlowEstimator(WBF_IM_Composite):
    def __init__(self):
        super().__init__(10, 10, *[PointEstimateScalarFieldIM(10, 10, default_value=1.0) for _ in range(3)])

    def proceed(self, delta_t):
        time.sleep(0.05)
        super().proceed(delta_t)


class TestEvaluation(unittest.TestCase):
    def test_a_slow_estimator_is_stopped(self):
        env = environment()
        observations = generate_observations({"kind": "random", "count": 30}, env, "Miniberry-10", 0)
        rows = evaluate(SlowEstimator(), env, observations, [10, 20, 30], stop_at=0.01)
        self.assertEqual({row["observations"] for row in rows}, {10})
        self.assertTrue(any(row["metric"] == "stopped" for row in rows))


if __name__ == "__main__":
    unittest.main()
