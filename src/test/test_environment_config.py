import pathlib
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from exp_run_config import Config
from information_model import im_score_weighted, im_score_weighted_asymmetric
from water_berry_farm import MiniberryFarm, WaterberryFarm, WBF_IM_DiskEstimator, WBF_Score_WeightedAsymmetric
from wbf_helper import cache_is_current, create_wbfe, environment_configuration, uses_cache


def environment_exp(directory, **values):
    """An environment exp with the shipped defaults, its data dir and exp/run file in the directory"""
    defaults = pathlib.Path(__file__).resolve().parents[2] / "data" / "expruns" / "environment" / "_defaults_environment.yaml"
    with open(defaults) as f:
        exp = yaml.safe_load(f)
    data_dir = pathlib.Path(directory, "data")
    data_dir.mkdir(exist_ok=True)
    exp |= {"typename": "Miniberry-10", "precompute-time": 5, "data_dir": str(data_dir),
            "exp_run_sys_indep_file": str(pathlib.Path(directory, "scenario.yaml"))}
    return exp | values


class TestPlanting(unittest.TestCase):
    def test_every_miniberry_cell_is_planted(self):
        for planting, expected in [("standard", {1, 2}), ("tomato-only", {2}), ("strawberry-only", {1})]:
            farm = MiniberryFarm(scale=3)
            farm.replant(planting)
            farm.create_type_map()
            self.assertEqual(set(np.unique(farm.type_map)), expected, planting)

    def test_tomato_only_waterberry_keeps_pond_and_wetland(self):
        farm = WaterberryFarm()
        farm.replant("tomato-only")
        types = {patch["name"]: patch["type"] for patch in farm.patches}
        self.assertNotIn("strawberry", types.values())
        self.assertEqual(types["strawberries"], "tomato")
        self.assertEqual((types["pond"], types["wetland buffer"]), ("pond", "wetland"))

    def test_unknown_planting_fails(self):
        with self.assertRaises(KeyError):
            MiniberryFarm().replant("potato-only")


class TestEnvironmentCreation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.directory.cleanup()

    def test_tomato_only_has_no_ccr_and_finite_scores(self):
        farm, environment = create_wbfe(environment_exp(self.directory.name, planting="tomato-only"))
        environment.proceed(3)
        self.assertFalse(environment.my_strawberry_mask.any())
        np.testing.assert_array_equal(environment.ccr.value, np.ones((10, 10)))
        score = WBF_Score_WeightedAsymmetric().score(environment, WBF_IM_DiskEstimator(10, 10))
        self.assertTrue(np.isfinite(score))

    def test_absent_crop_has_no_score_error(self):
        field = type("Field", (), {"value": np.ones((2, 2))})
        self.assertEqual(im_score_weighted(field, field, np.zeros((2, 2))), 0.0)
        self.assertEqual(im_score_weighted_asymmetric(field, field, 1.0, 10.0, np.zeros((2, 2))), 0.0)

    def test_parameters_change_the_spread(self):
        values = {}
        for p_transmission in [0.25, 0.9]:
            directory = pathlib.Path(self.directory.name, str(p_transmission))
            directory.mkdir()
            _, environment = create_wbfe(environment_exp(directory, **{"tylcv-p-transmission": p_transmission}))
            environment.proceed(4)
            values[p_transmission] = np.sum(environment.tylcv.value < 1.0)
        self.assertGreater(values[0.9], values[0.25])

    def test_initial_picture_starts_an_epidemic(self):
        picture = np.ones((10, 10))
        picture[7, 7] = 0.5  # one infected tomato cell
        plt.imsave(pathlib.Path(self.directory.name, "seed.png"), picture, cmap="gray", vmin=0, vmax=1)
        exp = environment_exp(self.directory.name, **{"tylcv-picture": "seed.png", "tylcv-picture-mode": "initial",
                                                       "tylcv-p-transmission": 0.9})
        farm, environment = create_wbfe(exp)
        environment.proceed(1)
        first = np.sum(environment.tylcv.value < 1.0)
        environment.proceed(3)
        self.assertGreater(np.sum(environment.tylcv.value < 1.0), first)
        # the strawberry cells are immune to TYLCV
        self.assertTrue(np.all(environment.tylcv.value[farm.type_map == farm.types["strawberry"]] == 1.0))

    def test_missing_picture_writes_a_template(self):
        exp = environment_exp(self.directory.name, **{"tylcv-picture": "new.png"})
        with self.assertRaisesRegex(Exception, "template"):
            create_wbfe(exp)
        self.assertTrue(pathlib.Path(self.directory.name, "new.png").exists())

    def test_replay_of_a_day_not_precomputed_fails(self):
        exp = environment_exp(self.directory.name)
        _, environment = create_wbfe(exp)
        environment.proceed(2)  # precomputes days 1 and 2
        _, replayed = create_wbfe(exp)
        replayed.proceed(2)
        np.testing.assert_array_equal(replayed.tylcv.value, environment.tylcv.value)
        with self.assertRaisesRegex(Exception, "not precomputed for day 3"):
            replayed.proceed(1)

    def test_environment_without_cache_is_computed_live(self):
        exp = environment_exp(self.directory.name, cache=False)
        fields = []
        for _ in range(2):
            _, environment = create_wbfe(exp)
            environment.proceed(3)
            fields.append(environment.soil.value)
        np.testing.assert_array_equal(fields[0], fields[1])
        self.assertEqual(list(pathlib.Path(exp["data_dir"]).iterdir()), [])

    def test_by_default_only_the_full_farm_is_not_cached(self):
        self.assertTrue(uses_cache({"cache": None, "typename": "Miniberry-100"}))
        self.assertFalse(uses_cache({"cache": None, "typename": "Waterberry"}))
        self.assertTrue(uses_cache({"cache": True, "typename": "Waterberry"}))


class TestCacheConfiguration(unittest.TestCase):
    def test_cache_is_current_only_for_the_same_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = environment_exp(directory, **{"tylcv-picture": "field.png"})
            plt.imsave(pathlib.Path(directory, "field.png"), np.ones((10, 10)), cmap="gray", vmin=0, vmax=1)
            values = dict(exp)
            experiment = type("Exp", (), {"values": values, "__getitem__": lambda self, key: values[key]})()
            saved = environment_configuration(experiment) | {Config.TIME_DONE: "now"}
            self.assertTrue(cache_is_current(experiment, saved))
            self.assertFalse(cache_is_current(experiment, {k: v for k, v in saved.items() if k != Config.TIME_DONE}))
            values["tylcv-p-transmission"] = 0.5
            self.assertFalse(cache_is_current(experiment, saved))
            values["tylcv-p-transmission"] = exp["tylcv-p-transmission"]
            plt.imsave(pathlib.Path(directory, "field.png"), np.zeros((10, 10)), cmap="gray", vmin=0, vmax=1)
            self.assertFalse(cache_is_current(experiment, saved))  # the picture changed
            self.assertFalse(cache_is_current(experiment, {k: v for k, v in saved.items() if k != "geometry-version"}))


if __name__ == "__main__":
    unittest.main()
