import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from papers.y2027_mrmr.exploration_package import ExplorationPackage


class TestExplorationPackage(unittest.TestCase):
    def test_lawnmowers_stay_inside_the_package(self):
        """An EP at the border of a 100x100 field: the last lawnmower row used to overshoot to y = 101"""
        for y_min, y_max in [(93, 99), (90, 99), (0, 5), (0, 6)]:
            ep = ExplorationPackage(80, 99, y_min, y_max, step=2)
            for lawnmower in [ep.lawnmower_horizontal_bottom_left, ep.lawnmower_horizontal_bottom_right,
                              ep.lawnmower_horizontal_top_left, ep.lawnmower_horizontal_top_right]:
                path = lawnmower()
                self.assertTrue((path[:, 0] >= 80).all() and (path[:, 0] <= 99).all(), lawnmower.__name__)
                self.assertTrue((path[:, 1] >= y_min).all() and (path[:, 1] <= y_max).all(),
                                (lawnmower.__name__, y_min, y_max))


if __name__ == "__main__":
    unittest.main()
