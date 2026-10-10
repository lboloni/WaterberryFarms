import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import time

import numpy as np

from papers.y2027_mrmr.exploration_package import ExplorationPackage, ExplorationPackageSet
from path_generators import get_path_length


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


def package_set(n, seed=0):
    rng = np.random.default_rng(seed)
    eps = ExplorationPackageSet()
    eps.ep_to_explore = []
    for _ in range(n):
        x, y = (int(v) for v in rng.integers(0, 90, 2))
        eps.ep_to_explore.append(ExplorationPackage(x, x + 8, y, y + 8, step=2))
    return eps


class TestShortestPath(unittest.TestCase):
    def test_small_searches_stay_exhaustive_and_optimal(self):
        """3 EPs: 6 * 64 = 384 combinations fit into the bound, so the result is the unbounded optimum"""
        eps = package_set(3)
        bounded, _ = eps.find_shortest_path_ep([0, 0], max_evaluations=2000)
        exhaustive, _ = eps.find_shortest_path_ep([0, 0], max_evaluations=None)
        self.assertEqual(get_path_length(bounded), get_path_length(exhaustive))

    def test_large_searches_are_bounded_complete_and_deterministic(self):
        """10 EPs: 10! * 4^10 combinations; the greedy path covers every EP, quickly, and the same way twice"""
        eps = package_set(10)
        start = time.time()
        path, ep_path = eps.find_shortest_path_ep([0, 0], end=[50, 50], max_evaluations=20000)
        self.assertLess(time.time() - start, 2.0)
        covered = [segment["ep"] for segment in ep_path if segment["ep"] is not None]
        self.assertEqual(sorted(map(id, covered)), sorted(map(id, eps.ep_to_explore)))
        self.assertEqual(list(path[0]), [0, 0])
        self.assertEqual(list(path[-1]), [50, 50])
        again, _ = eps.find_shortest_path_ep([0, 0], end=[50, 50], max_evaluations=20000)
        self.assertTrue(np.array_equal(path, again))


if __name__ == "__main__":
    unittest.main()
