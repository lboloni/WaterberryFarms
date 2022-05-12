from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import numpy as np
import unittest
import timeit
import pathlib 
import shutil

# allow the import from the source directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from WaterberryFarm import WaterberryFarm

import logging
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


class TestWaterberryGeometry(unittest.TestCase):
    """Tests for the FarmGeometry model, for specifically the WaterberryFarm setup."""

    def setUp(self):
        pass

    def test_positions(self):
        self.wbf = WaterberryFarm()
        self.assertFalse(self.wbf.point_in_component(3050, 50, "tomatoes"))
        self.assertTrue(self.wbf.point_in_component(1050, 2050, "strawberries"))
        self.assertTrue(self.wbf.point_in_component(2050, 2050, "pond"))
        self.assertTrue(self.wbf.point_in_component(1050, 1050, "strawberries"))
        self.assertTrue(self.wbf.point_in_component(3950, 3950, "wetland buffer"))
        self.assertFalse(self.wbf.point_in_component(2950, 2950, "tomatoes"))        

    def test_environment_scale(self):
        p = pathlib.Path.cwd()
        global savedir
        savedir = pathlib.Path(p.parent, "__Temporary", p.name + "_data", "unittest")
        savedir.mkdir(parents=True, exist_ok = True)

        for scale in [1, 5,10, 20, 40, 100, 200, 400]:
            print(scale)
            global wbf
            shutil.rmtree(savedir)
            savedir.mkdir(parents=True, exist_ok = True)
            time = timeit.timeit(f"global wbf; wbf = MiniberryFarm(scale={scale})", number=1,  globals=globals())
            print(f"MiniberryFarm scaled up {scale} times. Height: {wbf.height}, width: {wbf.width}")
            print(f"Creation: {time:0.2} seconds")
            time = timeit.timeit(f"wbf.create_type_map()", number=1,  globals=globals())
            print(f"Creation of the type map: {time:0.2} seconds")
            # creation of the environment
            wbfe = None
            time = timeit.timeit("global wbfe; wbfe = WaterberryFarmEnvironment(wbf, saved=False, seed = 10, savedir=savedir)", number=1,  globals=globals())
            print(f"Create WaterberryFarmEnvironment for it: {time:0.2} seconds")
            time = timeit.timeit(f"wbfe.proceed()", number=1,  globals=globals())
            print(f"Environment proceed: {time:0.2} seconds")

if __name__ == '__main__':
    unittest.main()