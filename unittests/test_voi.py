import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from information_model import ObservationRecord
from policy import FollowPathPolicy
from robot import Robot
from water_berry_farm import MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_DiskEstimator, WBF_Score_VoI, voi_credits
from wbf_simulate import simulate_1day


def field(values, uncertainty):
    return SimpleNamespace(value=np.array(values), uncertainty=np.array(uncertainty), default_value=1.0)


def credit(score, variant, robot):
    return sum(row["value"] for row in score["voi-credits"]
               if row["variant"] == variant and row["robot"] == robot)


class TestVoI(unittest.TestCase):
    def test_two_cell_worked_example(self):
        """The worked example of DESIGN-VOI.md, Section 4.1: alpha observes the infected cell A,
        and the estimator refit raises B to p_pos = 0.5"""
        env = SimpleNamespace(tylcv=SimpleNamespace(value=np.array([[0.5], [1.0]])),
                              ccr=SimpleNamespace(value=np.ones((2, 1))),
                              my_tomato_mask=np.array([[True], [True]]),
                              my_strawberry_mask=np.array([[False], [False]]))
        record = ObservationRecord(2, 1)
        im = SimpleNamespace(record=record, im_tylcv=field([[1.0], [1.0]], [[1.0], [1.0]]),
                             im_ccr=field([[1.0], [1.0]], [[1.0], [1.0]]))
        voi = WBF_Score_VoI(v_pos=100, v_neg=1, v_unknown=-3)
        baseline = voi.score(env, im)
        self.assertEqual(baseline["voi-expected"], 2)
        self.assertEqual(baseline["voi-ignorance"], -6)

        record.add({"x": 0, "y": 0, "robot": "alpha", "time": 0})
        im.im_tylcv = field([[0.5], [0.75]], [[0.0], [1.0]])
        event = voi.score(env, im)
        self.assertEqual(event["voi-expected"], 150.5)
        self.assertEqual(credit(event, "voi-expected", "alpha"), 99)
        self.assertEqual(credit(event, "voi-expected", None), 49.5)
        self.assertEqual(event["voi-ignorance"], 97)
        self.assertEqual(credit(event, "voi-ignorance", "alpha"), 103)
        self.assertEqual(credit(event, "voi-absolute", "alpha"), 100)

    def test_credits_add_up_in_a_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            farm = MiniberryFarm(scale=1)
            farm.create_type_map()
            environment = WaterberryFarmEnvironment(farm, use_saved=False, seed=10, savedir=directory)
            environment.proceed(6)
            robots = []
            for name, path in {"alpha": [[0, 5], [10, 5], [2, 5]], "beta": [[5, 0], [5, 10]],
                               "gamma": [[0, 1], [10, 9]]}.items():
                robot = Robot(name, path[0][0], path[0][1], 0)
                robot.assign_policy(FollowPathPolicy(1, path))
                robots.append(robot)
            estimator = WBF_IM_DiskEstimator(11, 11)
            baseline = WBF_Score_VoI().score(environment, WBF_IM_DiskEstimator(11, 11))
            results = simulate_1day(environment=environment, robots=robots, estimator=estimator,
                                    evaluator=WBF_Score_VoI(), timesteps=20, estimator_interval=3)

        # baseline + credits + update terms = the total at every scoring event
        for variant in WBF_Score_VoI.VARIANTS:
            credits = voi_credits(results, variant)
            for event in results["score-events"]:
                accumulated = sum(c[:event["timestep"] + 1].sum() for c in credits.values())
                self.assertAlmostEqual(baseline[variant] + accumulated, event["score"][variant])
        # absolute VoI is credited to the first observer of each cell, at its first timestep
        absolute = voi_credits(results, "voi-absolute")
        record = estimator.record
        mask = environment.my_tomato_mask | environment.my_strawberry_mask
        for name in ["alpha", "beta", "gamma"]:
            xs, ys = record.indices(robot=name)
            truth = np.where(np.where(environment.my_tomato_mask, environment.tylcv.value, environment.ccr.value) < 1.0, 100.0, 1.0)
            self.assertAlmostEqual(absolute[name].sum(), np.sum((mask * truth)[xs, ys]))
        self.assertEqual(absolute[None].sum(), 0)


if __name__ == "__main__":
    unittest.main()
