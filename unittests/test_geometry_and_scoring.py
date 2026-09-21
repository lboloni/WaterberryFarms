import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from environment import ScalarFieldEnvironment
from information_model import im_score_weighted, im_score_weighted_asymmetric
from water_berry_farm import MiniberryFarm, WaterberryFarm, WBF_MultiScore, WBF_Score_WeightedAsymmetric


def scalar(values):
    values = np.asarray(values, dtype=float)
    return ScalarFieldEnvironment("field", values.shape[0], values.shape[1], seed=0, value=values)


class TestGeometry(unittest.TestCase):
    def test_miniberry_type_map(self):
        farm = MiniberryFarm(scale=1)
        farm.create_type_map()
        self.assertEqual((farm.width, farm.height), (11, 11))
        self.assertEqual(farm.type_map[2, 2], farm.types["strawberry"])
        self.assertEqual(farm.type_map[2, 8], farm.types["tomato"])

    def test_waterberry_representative_components(self):
        farm = WaterberryFarm()
        self.assertEqual(farm.owner_area, [1000, 1000, 5000, 4000])
        self.assertTrue(farm.point_in_component(1050, 2050, "strawberries"))
        self.assertTrue(farm.point_in_component(2050, 2050, "pond"))
        self.assertTrue(farm.point_in_component(4500, 2000, "tomatoes"))
        self.assertTrue(farm.point_in_component(3950, 3950, "wetland buffer"))


class TestScoring(unittest.TestCase):
    def test_weighted_and_asymmetric_scores(self):
        env = scalar([[1.0, 0.0]])
        im = SimpleNamespace(value=np.array([[0.0, 1.0]]))
        mask = np.ones((1, 2))
        self.assertEqual(im_score_weighted(im, env, mask), -1.0)
        self.assertEqual(im_score_weighted_asymmetric(im, env, 1, 10, mask), -5.5)

    def test_masked_cells_do_not_contribute(self):
        env = scalar([[1.0, 0.0]])
        im = SimpleNamespace(value=np.array([[1.0, 1.0]]))
        mask = np.array([[1.0, 0.0]])
        self.assertEqual(im_score_weighted(im, env, mask), 0.0)

    def test_composite_and_multiscore_custom_scores_agree(self):
        env = SimpleNamespace(
            ccr=scalar([[1.0, 0.0]]),
            tylcv=scalar([[1.0, 0.0]]),
            soil=scalar([[0.0, 1.0]]),
            my_strawberry_mask=np.ones((1, 2)),
            my_tomato_mask=np.ones((1, 2)),
            my_soil_mask=np.ones((1, 2)))
        im = SimpleNamespace(
            im_ccr=SimpleNamespace(value=np.array([[0.0, 1.0]])),
            im_tylcv=SimpleNamespace(value=np.array([[0.0, 1.0]])),
            im_soil=SimpleNamespace(value=np.array([[1.0, 0.0]])))
        score = WBF_Score_WeightedAsymmetric(
            strawberry_negative_importance=3,
            tomato_negative_importance=7)
        multiscore = WBF_MultiScore(
            strawberry_negative_importance=3,
            tomato_negative_importance=7)
        self.assertAlmostEqual(score.score(env, im), multiscore.score(env, im)["custom"])


if __name__ == "__main__":
    unittest.main()
