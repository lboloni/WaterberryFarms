from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score_weighted, im_score_weighted_asymmetric
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import animation
from functools import partial
import numpy as np
import math
import bz2
import pathlib 
import pickle
import copy

import logging
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

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
        # update the width and height
        self.width = int(np.max([np.max(p["polygon"].get_xy()[:,0]) for p in self.patches])) + 1
        self.height = int(np.max([np.max(p["polygon"].get_xy()[:,1]) for p in self.patches])) + 1

    def visualize(self, ax):
        """Visualizes the farm geometry in an axis"""
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        for p in self.patches:
            # to allow visualization in multiple images...
            polygon = copy.deepcopy(p["polygon"])
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

    def create_type_map(self):
        """Calculates a map where each point shows what type of land is there."""
        self.type_map = np.zeros(self.width * self.height, dtype=np.int16)
        logging.info(f"create_type_map shape={self.type_map.shape}")
        Xpts = np.arange(0, self.width, 1, dtype=np.int16)
        Ypts = np.arange(0, self.height, 1, dtype=np.int16)
        X2D,Y2D = np.meshgrid(Xpts,Ypts, indexing="ij")
        points = np.column_stack((X2D.ravel(),Y2D.ravel()))
        for p in self.patches:  
            path = p["polygon"].get_path()
            type_value = self.types[p["type"]]
            # this radius thing apparently is necessary to deal with the uncertainty at the border
            marked = np.array(path.contains_points(points, radius=+0.5))
            self.type_map[marked] = type_value
        self.type_map = np.reshape(self.type_map, (self.width, self.height))

    def path_in_component(self, path, name):
        """Checks whether the specified path is in a given component. It assumes that components obscure each other, so if it is in a component that is above it, it will say no."""
        for p in reversed(self.patches):
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
        ## the owner's land (xmin, ymin, xmax, ymax)
        self.owner_area = [1000, 1000, 5000, 4000]
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
    """Implements the geometry of a small farm, for testing. Scalable size for testing performance. At scale 1, the size is 10x10"""

    def __init__(self, scale = 1):
        super().__init__()
        ## the owner's land (xmin, ymin, xmax, ymax)
        self.owner_area = [0, 0, 10 * scale, 10 * scale]
        ## the owner's patches
        # strawberry patch
        self.add_patch(name="strawberries", type="strawberry", area = [[3*scale,3*scale], [6* scale, 3*scale], [6*scale, 6*scale],[3*scale,6*scale] ], color="lightblue")
        # tomato patch
        self.add_patch(name="tomatoes", type="tomato", area=[[8*scale,8*scale], [10*scale, 8*scale], [10*scale, 10*scale],[8*scale, 10*scale] ], color="lightcoral")

class WaterberryFarmEnvironment(Environment):
    """An environment that describes the status of the soil and plant diseases on the Waterberry Farm"""

    def __init__(self, geometry, saved: bool,  seed, savedir):
        super().__init__(geometry.width, geometry.height, seed)
        self.geometry = geometry
        self.saved = saved        
        #
        # Tomato yellow leaf curl virus: epidemic spreading disease which spreads only on tomatoes
        #
        if self.saved:
            self.tylcv = PrecalculatedEnvironment(geometry.width, geometry.height, None, pathlib.Path(savedir, "precalc_tylcv"))
        else:
            # these are the parameters found in the ScenarioDesignWaterberryFarm
            parameters = {"infection_count": 3, "infection_value": 5, "proceed_count": 25, "p_transmission": 0.5, "infection_duration": 7}

            tylcv = EpidemicSpreadEnvironment("TYLCV", geometry.width, geometry.height, seed, p_transmission = parameters["p_transmission"], infection_duration = parameters["infection_duration"])
            # initialize the map for the ones which have tomatoes
            typecode = self.geometry.types["tomato"]
            # what I want here is to enumerate the indexes for which this is the value...
            mask = self.geometry.type_map == typecode
            tylcv.status[mask] = -2
            #self.tylcv.status[self.geometry.type_map[self.geometry.type_map == typecode]] = -2 

            for i in range(int(parameters["infection_count"])):
                locationx = int( tylcv.random.random(1) * tylcv.width )
                locationy = int ( tylcv.random.random(1) * tylcv.height )
                tylcv.status[locationx, locationy] = int(parameters["infection_value"])

            self.tylcv = PrecalculatedEnvironment(geometry.width, geometry.height, tylcv, pathlib.Path(savedir, "precalc_tylcv"))
        #
        # Charcoal Rot: epidemic spreading disease that spreads only on 
        #    strawberries
        #
        if self.saved:
            self.ccr = PrecalculatedEnvironment(geometry.width, geometry.height, None, pathlib.Path(savedir, "precalc_ccr"))
        else:
            ccr = EpidemicSpreadEnvironment("CCR", geometry.width, geometry.height, seed, p_transmission = 0.1, infection_duration = 10)
            typecode = self.geometry.types["strawberry"]
            ccr.status[self.geometry.type_map == typecode] = -2 
            self.ccr = PrecalculatedEnvironment(geometry.width, geometry.height, ccr, pathlib.Path(savedir, "precalc_ccr"))        #
        # Soil humidity: modeled with a dissipation model
        # 
        if self.saved:
            self.soil = PrecalculatedEnvironment(geometry.width, geometry.height, None, pathlib.Path(savedir, "precalc_soil"))
        else:
            soil = DissipationModelEnvironment("Soil humidity", geometry.width, geometry.height, seed, p_pollution = 0.1, pollution_max_strength= 1000, evolve_speed = 1)
            self.soil = PrecalculatedEnvironment(geometry.width, geometry.height, soil, pathlib.Path(savedir, "precalc_soil"))

        self.my_owner_mask = np.full((self.width, self.height), False)
        self.my_owner_mask[self.geometry.owner_area[0]:self.geometry.owner_area[2], self.geometry.owner_area[1]:self.geometry.owner_area[3]] = True

        self.my_strawberry_mask = np.equal(self.geometry.type_map, self.geometry.types["strawberry"])
        self.my_strawberry_mask = np.logical_and(self.my_strawberry_mask, self.my_owner_mask)
        self.my_tomato_mask = np.equal(self.geometry.type_map, self.geometry.types["tomato"])
        self.my_tomato_mask = np.logical_and(self.my_tomato_mask, self.my_owner_mask)
        self.my_soil_mask = np.ones((self.width, self.height))
        self.my_soil_mask = np.logical_and(self.my_soil_mask, self.my_owner_mask)



    def inner_proceed(self, delta_t = 1.0):        
        self.tylcv.proceed(delta_t)
        self.ccr.proceed(delta_t)
        self.soil.proceed(delta_t)

    def get_observation(self, pos):
        """For a given position and time creates a composite observation. The format for pos is [x,y,time]"""
        obs = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}

        tylcv = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1],StoredObservationIM.TIME: pos[2]}
        tylcv[StoredObservationIM.VALUE] = self.tylcv.get(pos[0], pos[1])
        obs["TYLCV"] = tylcv

        ccr = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}
        ccr[StoredObservationIM.VALUE] = self.ccr.get(pos[0], pos[1])
        obs["CCR"] = ccr

        soil = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}
        soil[StoredObservationIM.VALUE] = self.soil.get(pos[0], pos[1])
        obs["Soil"] = soil

        return obs

    def visualize(self, fig = None, ax_geom = None, ax_tylcv= None, ax_ccr = None, ax_soil = None):
        """Plot the components into a four panel figure. """
        if fig == None:
            self.fig, ((self.ax_geom, self.ax_tylcv), (self.ax_ccr, self.ax_soil)) = plt.subplots(2,2)
        else:
            self.fig, self.ax_geom, self.ax_tylcv, self.ax_ccr, self.ax_soil = fig, ax_geom,  ax_tylcv, ax_ccr, ax_soil
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

class WaterberryFarmInformationModel(StoredObservationIM):
    """The information model for the waterberry farm. It is basically a collection of three information models, one for each environmental model."""

    def __init__(self, name, width, height):
        """Creating the stuff"""
        super().__init__(name, width, height)
        self.gp = False
        if self.gp:
            self.im_tylcv = GaussianProcessScalarFieldIM("TYLCV", width, height)
            self.im_ccr = GaussianProcessScalarFieldIM("CCR", width, height)
            self.im_soil = GaussianProcessScalarFieldIM("Soil",width, height)
        else:
            self.im_tylcv = DiskEstimateScalarFieldIM("TYLCV", width, height, disk_radius=None)
            self.im_ccr = DiskEstimateScalarFieldIM("CCR", width, height, disk_radius=None)
            self.im_soil = DiskEstimateScalarFieldIM("Soil",width, height, disk_radius=None)

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


def waterberry_score(env: WaterberryFarmEnvironment, im: WaterberryFarmInformationModel):
    """A specialized score function for the waterberry farm environment, which is individually weights the values of the different components
    """
    params = {}
    params["strawberry_importance"] = 0.2 # not lead to full crop loss
    params["strawberry_negative_importance"] = 10 # false negatives are more important
    params["tomato_importance"] = 1.0 # leads to full crop loss
    params["tomato_negative_symmetry"] = 5 # false negatives are more important
    soil_importance = 0.1 # the exact value is of lower importance on revenue

    score = 0
    score +=  params["strawberry_importance"] * im_score_weighted_asymmetric(env.ccr, im.im_ccr, 1.0, params["strawberry_negative_importance"], env.my_strawberry_mask)

    score += params["tomato_importance"] * im_score_weighted_asymmetric(env.tylcv, im.im_tylcv, 1.0, params["strawberry_negative_importance"], env.my_tomato_mask)

    score += soil_importance * im_score_weighted(env.soil, im.im_soil, env.my_soil_mask)
    return score


def create_wbfe(saved: bool, wbf_prec = None, typename = "Miniberry-10"):
    """Helper function for the creation of a waterberry farm environment on which we can run experiments. It performs a caching process, if the files already exists, it just reloads them. This will save time for expensive simulations."""
    # p = pathlib.Path.cwd()
    p = pathlib.Path(__file__).parent.resolve().parent.parent
    savedir = pathlib.Path(p.parent, "__Temporary", p.name + "_data", typename)
    savedir.mkdir(parents=True, exist_ok = True)
    path_geometry = pathlib.Path(savedir,"farm_geometry")
    path_environment = pathlib.Path(savedir,"farm_environment")
    if path_geometry.exists():
        logging.info("loading the geometry and environment from saved data")
        with bz2.open(path_geometry, "rb") as f:
            wbf = pickle.load(f)
        #with open(path_environment, "rb") as f:
        #    wbfe = pickle.load(f)
        #    wbfe.saved = saved
        logging.info("loading done")
        wbfe = WaterberryFarmEnvironment(wbf, saved, seed = 10, savedir = savedir)
        return wbf, wbfe, savedir
    if wbf_prec == None:
        if typename == "Miniberry-10":
            wbf = MiniberryFarm(scale=1)
        elif typename == "Miniberry-30":
            wbf = MiniberryFarm(scale=3)
        elif typename == "Miniberry-100":
            wbf = MiniberryFarm(scale=10)
        elif typename == "Waterberry":            
            wbf = WaterberryFarm()
        else:
            raise Exception(f"Unknown type {typename}")
        # wbf = MiniberryFarm(scale=10)
    else:
        wbf = wbf_prec
    wbf.create_type_map()
    wbfe = WaterberryFarmEnvironment(wbf, saved, seed = 10, savedir = savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    with bz2.open(path_geometry, "wb") as f:
        pickle.dump(wbf, f)
    with bz2.open(path_environment, "wb") as f:
        pickle.dump(wbfe, f)
    return wbf, wbfe, savedir
