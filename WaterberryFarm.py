import Environment
from InformationModel import InformationModel, ScalarFieldInformationModel_stored_observation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import animation
from functools import partial
import numpy as np
import math
import unittest
import timeit

import logging
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.info("Hello world")

class FarmGeometry:
    """Patches should be added in decreasing order"""

    def __init__(self):
        self.patches = []
        # every type is associated with a number
        self.types = {}

    def add_patch(self, name, type, area, color):
        """Adds a new patch of the specified name, type (strawberry, tomato, water), area boundaries and color"""
        patch = {}
        patch["name"] = name
        patch["type"] = type 
        # if we knew this type, ok, if not find a number for it
        if type not in self.types:
            self.types[type] = len(self.types) + 1
        patch["area"] = area
        patch["color"] = color
        patch["polygon"] = Polygon(area, color=color)
        self.patches.append(patch)
        # determine the sizes xmin and ymin should be zero        
        # FIXME: we don't need to iterate here, just min with the current one
        #xmin = np.min([np.min(p["polygon"].get_xy()[:,0]) for p in self.patches])
        self.width = int(np.max([np.max(p["polygon"].get_xy()[:,0]) for p in self.patches])) + 1
        #ymin = np.min([np.min(p["polygon"].get_xy()[:,1]) for p in self.patches])
        self.height = int(np.max([np.max(p["polygon"].get_xy()[:,1]) for p in self.patches])) + 1

    def visualize(self, ax):
        """Visualizes the patch in an axis"""
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        for p in self.patches:
            polygon = p["polygon"]
            center = np.mean(polygon.get_xy(),0)
            patch = ax.add_patch(polygon)
            ax.text(center[0],center[1],p["name"])

    def point_in_component(self, x,y, name):
        """Checks whether location x, y is in a given component. It assumes that components obscure each other, so if it is in a component that is above it, it will say no."""
        for p in reversed(self.patches):
            # path = p["area"].get_path()
            if p["polygon"].get_path().contains_point([x,y]):
                if p["name"] == name:
                    return True
                else:
                    return False
        return False

    def point_type(self, x, y):
        """Returns the type of component here. Useful for setting masks. 
        FIXME: this is not vectorized, it might be slow, might be worth to make a typemap"""
        for p in reversed(self.patches):
            if p["polygon"].get_path().contains_point([x,y]):
                return p["type"]

    def create_type_map(self):
        """Implementation with reshaping..."""
        self.type_map = np.zeros(self.width * self.height, dtype=np.int16)
        print(self.type_map.shape)
        Xpts = np.arange(0, self.width, 1, dtype=np.int16)
        Ypts = np.arange(0, self.height, 1, dtype=np.int16)
        X2D,Y2D = np.meshgrid(Xpts,Ypts, indexing="ij")
        points = np.column_stack((X2D.ravel(),Y2D.ravel()))
        for p in self.patches:  
            path = p["polygon"].get_path()
            type_value = self.types[p["type"]]
            # print(type_value)
            # this radius thing apparently is necessary to deal with the uncertainty at the border
            marked = np.array(path.contains_points(points, radius=+0.5))
            # print(marked)
            self.type_map[marked] = type_value
        self.type_map = np.reshape(self.type_map, (self.width, self.height))

    def path_in_component(self, path, name):
        """Checks whether the specified path is in a given component. It assumes that components obscure each other, so if it is in a component that is above it, it will say no."""
        for p in reversed(self.patches):
            # path = p["area"].get_path()
            if p["polygon"].get_path().contains_path(path):
                if p["name"] == name:
                    return True
                else:
                    return False
        return False

    def list_components(self):
        return [p["name"] for p in self.patches]

class WaterberryFarm(FarmGeometry):
    """Implements the geometry of WaterberryFarm"""

    def __init__(self):
        super().__init__()

        ## the owner's patches
        # strawberry patch
        self.add_patch(name="strawberries", type="strawberry", area = [[1000,1000], [4000, 1000], [3000, 4000],[1000, 4000] ], color="lightblue")
        # tomato patch
        self.add_patch(name="tomatoes", type="tomato", area=[[4000,1000], [5000, 1000], [5000, 4000],[3000, 4000] ], color="lightcoral")
        # pond area: manually set up as a ellipse and creates as a polygon
        areapond = [[2000 + 200 * math.sin(theta), 2000 + 200 * math.cos(theta)] for theta in [math.pi * i / 12 for i in range(24)]]
        self.add_patch(name="pond", type="pond", area=areapond, color="blue")
        # wetland buffer 
        areawetland = [[4000 + 500 * math.sin(theta), 4000 + 500 * math.cos(theta)] for theta in [math.pi * i / 12 for i in range(6, 19)]]
        self.add_patch(name="wetland buffer", type="wetland", area=areawetland, color="green")

        ## the neighbors patches
        self.add_patch(name="N1-strawberries", type="strawberry", area = [[0,0], [0, 1000], [6000, 1000],[6000, 0] ], color="azure")
        self.add_patch(name="N2-strawberries", type="strawberry", area = [[5000,1000], [6000, 1000], [6000, 4000],[5000, 4000] ], color="azure")
        self.add_patch(name="N3-tomatoes", type="tomato", area = [[6000,4000], [6000, 5000], [0, 5000],[0, 4000] ], color="mistyrose")
        self.add_patch(name="N4-tomatoes", type="tomato", area = [[0,4000], [1000, 4000], [1000, 1000],[0, 1000] ], color="mistyrose")

class MiniberryFarm(FarmGeometry):
    """Implements the geometry of a small farm, for testing. Scalable size for testing performance"""

    def __init__(self, scale = 1):
        super().__init__()
        print(6*scale)        
        ## the owner's patches
        # strawberry patch
        self.add_patch(name="strawberries", type="strawberry", area = [[3*scale,3*scale], [6* scale, 3*scale], [6*scale, 6*scale],[3*scale,6*scale] ], color="lightblue")
        # tomato patch
        self.add_patch(name="tomatoes", type="tomato", area=[[8*scale,8*scale], [10*scale, 8*scale], [10*scale, 10*scale],[8*scale, 10*scale] ], color="lightcoral")

class WaterberryFarmEnvironment(Environment.Environment):
    """An environment that describes the status of the soil and plant diseases on the Waterberry Farm"""

    def __init__(self, geometry, seed):
        super().__init__(geometry.width, geometry.height, seed)
        self.geometry = geometry
        #
        # Tomato yellow leaf curl virus: epidemic spreading disease which spreads only on tomatoes
        #
        self.tylcv = Environment.EpidemicSpreadEnvironment("TYLCV", geometry.width, geometry.height, seed, p_transmission = 0.2, infection_duration = 5)
        # initialize the map for the ones which have tomatoes
        typecode = self.geometry.types["tomato"]
        # what I want here is to enumerate the indexes for which this is the value...
        mask = self.geometry.type_map == typecode
        self.tylcv.status[mask] = -2
        #self.tylcv.status[self.geometry.type_map[self.geometry.type_map == typecode]] = -2 
        #
        # Charcoal Rot: epidemic spreading disease that spreads only on 
        #    strawberries
        #
        self.ccr = Environment.EpidemicSpreadEnvironment("CCR", geometry.width, geometry.height, seed, p_transmission = 0.1, infection_duration = 10)
        typecode = self.geometry.types["strawberry"]
        self.ccr.status[self.geometry.type_map == typecode] = -2 
        #
        # Soil humidity: modeled with a dissipation model
        # 
        self.soil = Environment.DissipationModelEnvironment("Soil humidity", geometry.width, geometry.height, seed, p_pollution = 0.1, pollution_max_strength= 1000, evolve_speed = 1)

    def proceed(self, delta_t = 1.0):        
        self.tylcv.proceed(delta_t)
        self.ccr.proceed(delta_t)
        self.soil.proceed(delta_t)

    def visualize(self):
        """Plot the components into a four panel figure"""
        self.fig, ((self.ax_geom, self.ax_tylcv), (self.ax_ccr, self.ax_soil)) = plt.subplots(2,2)
        self.geometry.visualize(self.ax_geom)
        self.ax_geom.set_title("Layout")
        self.image_tylcv = self.ax_tylcv.imshow(self.tylcv.value.T, vmin=0, vmax=1, origin="lower")
        # ax_tylcv.imshow(self.tylcv.status.T, origin="lower")
        self.ax_tylcv.set_title("TYLCV")
        self.image_ccr = self.ax_ccr.imshow(self.ccr.value.T, vmin=0, vmax=1, origin="lower")
        self.ax_ccr.set_title("CCR")
        self.image_soil = self.ax_soil.imshow(self.soil.value.T, vmin=0, vmax=1, origin="lower")
        self.ax_soil.set_title("Soil")

    def animate(self, i):
        """Animates an environment by letting the environment proceed a timestep, then setting the values into image."""
        logging.info(f"animate {i} before proceed")        
        self.proceed(1.0)
        logging.info(f"animate {i} after proceed")        

        v = self.tylcv.value.copy()
        self.image_tylcv.set_array(v.T)

        v = self.ccr.value.copy()
        self.image_ccr.set_array(v.T)

        v = self.soil.value.copy()
        self.image_soil.set_array(v.T)
        logging.info(f"animate {i} after setting arrays")        

        return [self.image_tylcv, self.image_ccr, self.image_soil]

    def animate_environment(self):
        #fig, ax = plt.subplots()
        # im  = ax.imshow(env.value, cmap="gray", vmin=0, vmax=1.0)
        #axesimage  = ax.imshow(env.value, vmin=0, vmax=1)
        anim = animation.FuncAnimation(self.fig, partial(self.animate), frames=100, interval=50, blit=True)
        return anim

class WaterberryFarmInformationModel(InformationModel):
    """The information model for the waterberry farm. It is basically a collection of three information models, one for each environmental model."""

    def __init__(self, name, width, height):
        """Creating the stuff"""
        super().__init__(name, width, height)
        self.im_tylcv = ScalarFieldInformationModel_stored_observation("TYLCV", width, height)
        self.im_ccr = ScalarFieldInformationModel_stored_observation("CCR", width, height)
        self.im_soil = ScalarFieldInformationModel_stored_observation("Soil")

    def add_observation(self, observation: dict):
        """It assumes that the observation is a dictionary with the components being the individual observations for TYLCV, CCR and soil humidity. 
        This function just distributes the components of the observation to the subcomponents"""
        self.im_tylcv.add_observation(observation["TYLCV"])
        self.im_ccr.add_observation(observation["CCR"])
        self.im_soil.add_observation(observation["Soil"])

    def proceed(self, delta_t: float):
        """It calls the proceed functions that call the estimates for the individual subcomponents"""
        self.im_tylcv.proceed(delta_t)
        self.im_ccr.proceed(delta_t)
        self.im_soil.proceed(delta_t)

    def visualize(self):
        """Visualize the estimates and the uncertainty models for the three components"""
        pass


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
        for scale in [1, 5,10, 20, 40, 100, 200, 400]:
            print(scale)
            global wbf
            time = timeit.timeit(f"global wbf; wbf = MiniberryFarm(scale={scale})", number=1,  globals=globals())
            print(f"MiniberryFarm scaled up {scale} times. Height: {wbf.height}, width: {wbf.width}")
            print(f"Creation: {time:0.2} seconds")
            time = timeit.timeit(f"wbf.create_type_map()", number=1,  globals=globals())
            print(f"Creation of the type map: {time:0.2} seconds")
            # creation of the environment
            wbfe = None
            time = timeit.timeit("global wbfe; wbfe = WaterberryFarmEnvironment(wbf, seed = 10)", number=1,  globals=globals())
            print(f"Create WaterberryFarmEnvironment for it: {time:0.2} seconds")
            time = timeit.timeit(f"wbfe.proceed()", number=1,  globals=globals())
            print(f"Environment proceed: {time:0.2} seconds")



if __name__ == "__main__":
    if True:
        unittest.main()
    if False:
        a = np.array([1, 2, 3, 4, 5, 6])
        index = np.array([1, 2, 4])
        mask = np.array([False, True, True])
        # index[mask] = 0
        print(index[mask])
        print(a[index[mask]])
        #a[[1, 2, 4]] = 0
        #print(a)
    if False:
        logging.basicConfig(level=logging.DEBUG)
        # logging.setLevel(logging.INFO)
        wbf = WaterberryFarm()
        # wbf = MiniberryFarm(scale=400)
        wbf.create_type_map()
        wbfe = WaterberryFarmEnvironment(wbf, seed = 10)

        # place some infections
        for i in range(1000):
            locationx = int( wbfe.tylcv.random.random(1) * wbfe.tylcv.width )
            locationy = int ( wbfe.tylcv.random.random(1) * wbfe.tylcv.height )
            wbfe.tylcv.status[locationx, locationy] = 5
            # wbfe.ccr.status[1050,1050] = 5

        for i in range(1):
            wbfe.proceed()
        wbfe.visualize()
        logging.info("Visualize done, proceed to animate")
        # This is working at approximately 2 sec per frame for the environment progress. Not very fast, but it should be roughly ok. 
        anim = wbfe.animate_environment()
        plt.show()
    if False:
        Ypts = np.arange(1, 4000, 1)
        Xpts = np.arange(1, 3000, 1)
        X2D,Y2D = np.meshgrid(Ypts,Xpts)
        points = np.column_stack((Y2D.ravel(),X2D.ravel()))
        print(points)