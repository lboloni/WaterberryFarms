import logging
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from disease_maps import cluster_stats, diseased_count, generate_disease_map
from wbf_helper import create_wbfe

logging.getLogger().setLevel(logging.WARNING)

DEFAULTS = pathlib.Path(__file__).resolve().parents[2] / "data" / "expruns" / "environment" / "_defaults_environment.yaml"


class TestGeneratedDiseaseMaps(unittest.TestCase):
    def test_pairs_have_exactly_the_target_number_of_diseased_plants(self):
        for size in [30, 60]:
            for version in ["clustered", "unclustered"]:
                env, _ = generate_disease_map(size, size, version, seed=1, fraction=0.05)
                self.assertEqual(diseased_count(env.value), int(0.05 * size * size), (size, version))

    def test_clustered_maps_have_fewer_and_larger_spots(self):
        clustered, _ = generate_disease_map(60, 60, "clustered", seed=1, fraction=0.05)
        unclustered, _ = generate_disease_map(60, 60, "unclustered", seed=1, fraction=0.05)
        c, u = cluster_stats(clustered.value), cluster_stats(unclustered.value)
        self.assertLess(c["spots"], u["spots"])
        self.assertGreater(c["mean spot"], u["mean spot"])

    def test_maps_are_determined_by_the_seed(self):
        for version in ["clustered", "unclustered"]:
            first, _ = generate_disease_map(40, 40, version, seed=7, fraction=0.05)
            again, _ = generate_disease_map(40, 40, version, seed=7, fraction=0.05)
            other, _ = generate_disease_map(40, 40, version, seed=8, fraction=0.05)
            self.assertTrue(np.array_equal(first.value, again.value))
            self.assertFalse(np.array_equal(first.value, other.value))

    def test_immune_cells_stay_healthy_and_the_target_is_relative_to_the_plantable_cells(self):
        mask = np.zeros((40, 40))
        mask[:, :20] = -2  # half of the field is immune
        for version in ["clustered", "unclustered"]:
            env, _ = generate_disease_map(40, 40, version, seed=1, fraction=0.05, immunity_mask=mask)
            self.assertEqual(diseased_count(env.value), int(0.05 * 800))
            self.assertTrue(np.all(env.value[:, :20] == 1.0))


class TestGeneratedEnvironment(unittest.TestCase):
    def environment_exp(self, **values):
        with open(DEFAULTS) as f:
            exp = yaml.safe_load(f)
        exp.update({"typename": "Miniberry-30", "precompute-time": 5, "planting": "tomato-only",
                    "cache": False, "data_dir": tempfile.mkdtemp()}, **values)
        return exp

    def test_the_generated_map_is_the_field_on_every_day(self):
        exp = self.environment_exp(**{"tylcv-generated": "clustered", "tylcv-generated-seed": 3})
        _, wbfe = create_wbfe(exp)
        expected, _ = generate_disease_map(30, 30, "clustered", seed=3, fraction=0.05,
                                           immunity_mask=np.zeros((30, 30)))
        wbfe.proceed(1)
        first = np.copy(wbfe.tylcv.value)
        wbfe.proceed(10)
        self.assertTrue(np.array_equal(first, expected.value))
        self.assertTrue(np.array_equal(wbfe.tylcv.value, expected.value))

    def test_picture_and_generated_map_are_exclusive(self):
        exp = self.environment_exp(**{"tylcv-generated": "clustered", "tylcv-picture": "field.png"})
        with self.assertRaisesRegex(Exception, "exclusive"):
            create_wbfe(exp)


if __name__ == "__main__":
    unittest.main()
