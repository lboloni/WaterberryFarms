import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from path_generators import find_fixed_budget_lawnmower, find_fixed_budget_spiral, generate_lawnmower, generate_spiral_path, get_path_length
from papers.y2023_glr.grid_limited_randomness import GLR_end


class TestPathGenerators(unittest.TestCase):
    def test_path_length(self):
        self.assertEqual(get_path_length([[3, 4], [6, 8]], [0, 0]), 10)

    def test_lawnmower_bounds(self):
        path = generate_lawnmower(2, 8, 3, 9, winds=3)
        self.assertTrue(np.all((path[:, 0] >= 2) & (path[:, 0] <= 8)))
        self.assertTrue(np.all((path[:, 1] >= 3) & (path[:, 1] <= 9)))

    def test_fixed_budget_lawnmower_is_maximal_and_feasible(self):
        budget = 80
        path = find_fixed_budget_lawnmower([0, 0], 0, 10, 0, 10, 1, budget)
        self.assertLessEqual(get_path_length(path, [0, 0]), budget)
        winds = (len(path) - 2) // 4
        next_path = generate_lawnmower(0, 10, 0, 10, winds + 1)
        self.assertGreater(get_path_length(next_path, [0, 0]), budget)

    def test_fixed_budget_lawnmower_rejects_impossible_budget(self):
        with self.assertRaisesRegex(Exception, "budget is too small"):
            find_fixed_budget_lawnmower([0, 0], 0, 10, 0, 10, 1, 1)

    def test_spiral_bounds_and_budget(self):
        path = generate_spiral_path(0, 10, 0, 10, step=2)
        self.assertTrue(np.all((path[:, 0] >= 0) & (path[:, 0] <= 10)))
        self.assertTrue(np.all((path[:, 1] >= 0) & (path[:, 1] <= 10)))
        budget_path = find_fixed_budget_spiral([0, 0], 0, 10, 0, 10, 1, 60)
        self.assertLessEqual(get_path_length(budget_path, [0, 0]), 60)

    def test_glr_is_reproducible(self):
        geometry = {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10, "velocity": 1}
        path1 = GLR_end([], [0, 0], geometry, time=30, seed=4, h_cells=2, v_cells=2)
        path2 = GLR_end([], [0, 0], geometry, time=30, seed=4, h_cells=2, v_cells=2)
        np.testing.assert_array_equal(path1, path2)


if __name__ == "__main__":
    unittest.main()
