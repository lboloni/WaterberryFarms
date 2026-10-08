from exp_run_config import Config
from environment import Environment, EpidemicSpreadEnvironment, PrecalculatedEnvironment, SoilMoistureEnvironment
from information_model import StoredObservationIM, ObservationRecord, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score_weighted, im_score_weighted_asymmetric, im_score, im_score_rmse
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
import sys

import logging
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


class FarmGeometry:
    """A farm as a list of patches, added from the bottom up (later patches cover earlier ones). 
    Coordinates are continuous: the cell (x, y) of the grid is the unit square [x, x+1) x [y, y+1), so the 
    polygon vertices are on cell boundaries, and a farm whose patches reach x = 10 is 10 cells wide."""

    # incremented when create_type_map changes the type map of existing geometries (invalidates the caches)
    RASTERIZATION_VERSION = 3

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
        # update the width and height: the number of cells covered by the patches
        self.width = int(math.ceil(
            np.max([np.max(p["polygon"].get_xy()[:, 0]) for p in self.patches])))
        self.height = int(math.ceil(
            np.max([np.max(p["polygon"].get_xy()[:, 1]) for p in self.patches])))

    def visualize(self, ax):
        """Visualizes the farm geometry in an axis"""
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        for p in self.patches:
            # to allow visualization in multiple images...
            polygon = copy.deepcopy(p["polygon"])
            center = np.mean(polygon.get_xy(), 0)
            patch = ax.add_patch(polygon)
            ax.text(center[0], center[1], p["name"])

    def point_in_component(self, x, y, name):
        """Checks whether location x, y is in a given component. It assumes that components obscure each other, so if it is in a component that is above it, it will say no."""
        for p in reversed(self.patches):
            # path = p["area"].get_path()
            if p["polygon"].get_path().contains_point([x, y]):
                if p["name"] == name:
                    return True
                else:
                    return False
        return False

    PLANTINGS = {"standard": {}, "tomato-only": {"strawberry": "tomato"}, "strawberry-only": {"tomato": "strawberry"}}

    def replant(self, planting):
        """Changes the crops of the patches according to the planting (standard, tomato-only or 
        strawberry-only). Must be called before create_type_map."""
        for patch in self.patches:
            patch["type"] = self.PLANTINGS[planting].get(patch["type"], patch["type"])

    def create_type_map(self):
        """Calculates a map where each cell shows what type of land is there: the type of the last patch 
        that contains the center of the cell (0 if none)."""
        self.type_map = np.zeros(self.width * self.height, dtype=np.int16)
        logging.info(f"create_type_map shape={self.type_map.shape}")
        X2D, Y2D = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing="ij")
        centers = np.column_stack((X2D.ravel(), Y2D.ravel())) + 0.5
        for p in self.patches:
            marked = np.array(p["polygon"].get_path().contains_points(centers))
            self.type_map[marked] = self.types[p["type"]]
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
        # the owner's land (xmin, ymin, xmax, ymax)
        self.owner_area = [1000, 1000, 5000, 4000]
        # the owner's patches
        # strawberry patch
        self.add_patch(name="strawberries", type="strawberry", area=[[1000, 1000], [
                       4000, 1000], [3000, 4000], [1000, 4000]], color="lightblue")
        # tomato patch
        self.add_patch(name="tomatoes", type="tomato", area=[[4000, 1000], [
                       5000, 1000], [5000, 4000], [3000, 4000]], color="lightcoral")
        # pond area: manually set up as a ellipse and creates as a polygon
        areapond = [[2000 + 200 * math.sin(theta), 2000 + 200 * math.cos(theta)]
                    for theta in [math.pi * i / 12 for i in range(24)]]
        self.add_patch(name="pond", type="pond", area=areapond, color="blue")
        # wetland buffer
        areawetland = [[4000 + 500 * math.sin(theta), 4000 + 500 * math.cos(
            theta)] for theta in [math.pi * i / 12 for i in range(6, 19)]]
        self.add_patch(name="wetland buffer", type="wetland",
                       area=areawetland, color="green")

        # the neighbors patches
        self.add_patch(name="N1-strawberries", type="strawberry",
                       area=[[0, 0], [0, 1000], [6000, 1000], [6000, 0]], color="azure")
        self.add_patch(name="N2-strawberries", type="strawberry",
                       area=[[5000, 1000], [6000, 1000], [6000, 4000], [5000, 4000]], color="azure")
        self.add_patch(name="N3-tomatoes", type="tomato",
                       area=[[6000, 4000], [6000, 5000], [0, 5000], [0, 4000]], color="mistyrose")
        self.add_patch(name="N4-tomatoes", type="tomato",
                       area=[[0, 4000], [1000, 4000], [1000, 1000], [0, 1000]], color="mistyrose")


class MiniberryFarm(FarmGeometry):
    """Implements the geometry of a small farm, for testing. Scalable size for testing performance. At scale 1, the size is 10x10 cells"""

    def __init__(self, scale=1):
        super().__init__()
        # the owner's land (xmin, ymin, xmax, ymax)
        self.owner_area = [0, 0, 10 * scale, 10 * scale]
        # the owner's patches
        # strawberry patch
        #self.add_patch(name="strawberries", type="strawberry", area = [[3*scale,3*scale], [6* scale, 3*scale], [6*scale, 6*scale],[3*scale,6*scale] ], color="lightblue")
        self.add_patch(name="strawberries", type="strawberry", area=[[0, 0], [
                       0, 5 * scale], [10 * scale, 5 * scale], [10 * scale, 0]], color="lightblue")
        # tomato patch
        # self.add_patch(name="tomatoes", type="tomato", area=[[8*scale,8*scale], [10*scale, 8*scale], [10*scale, 10*scale],[8*scale, 10*scale] ], color="lightcoral")
        self.add_patch(name="tomatoes", type="tomato", area=[
                       [0, 5*scale], [0, 10*scale], [10*scale, 10*scale], [10*scale, 5*scale]], color="lightcoral")


def default_infection_seeds(width):
    """The number of initial infections of an epidemic, scaled with the width of the grid"""
    return 3 * max(int(width / 30), 1)


def default_spread_dimension(width):
    """The size of the neighborhood through which an epidemic spreads, scaled with the width of the grid"""
    return min(11, int(math.sqrt(width) / 6) * 2 + 3)


class WaterberryFarmEnvironment(Environment):
    """An environment that describes the status of the soil and plant diseases on the Waterberry Farm"""

    def __init__(self, geometry, use_saved: bool, seed, savedir, tylcv = None, ccr = None, soil = None):
        """Create the environment. 
        use_saved: if true, the fields are replayed from the values precomputed into savedir. If false, 
        the field models are run, and their values are saved into savedir, or not saved if savedir is None.
        tylcv, ccr, soil: the parameters of the field models; the default values were calibrated in 
        ScenarioDesignWaterberryFarm. A None infection_seeds or spread_dimension is scaled with the grid."""
        super().__init__(geometry.width, geometry.height, seed)
        self.geometry = geometry
        self.use_saved = use_saved
        if self.use_saved:
            self.tylcv = self.field("tylcv", None, savedir)
            self.ccr = self.field("ccr", None, savedir)
            self.soil = self.field("soil", None, savedir)
        else:
            # Tomato yellow leaf curl virus: epidemic spreading disease which spreads only on tomatoes
            self.tylcv = self.field("tylcv", self.create_epidemic("TYLCV", "tomato", seed,
                {"p_transmission": 0.25, "infection_duration": 5, "infection_seeds": None, 
                 "spread_dimension": None} | (tylcv or {})), savedir)
            # Charcoal Rot: epidemic spreading disease that spreads only on strawberries
            self.ccr = self.field("ccr", self.create_epidemic("CCR", "strawberry", seed,
                {"p_transmission": 0.15, "infection_duration": 10, "infection_seeds": None, 
                 "spread_dimension": None} | (ccr or {})), savedir)
            # Soil moisture model
            soil = {"seed": 1, "evaporation": 0.04, "rainfall": 0.06, "rain_likelihood": 0.8} | (soil or {})
            self.soil = self.field("soil", SoilMoistureEnvironment("soil", geometry.width, geometry.height, 
                seed=soil["seed"], evaporation=soil["evaporation"], rainfall=soil["rainfall"], 
                rain_likelihood=soil["rain_likelihood"]), savedir)

        self.my_owner_mask = np.full((self.width, self.height), False)
        self.my_owner_mask[self.geometry.owner_area[0]:self.geometry.owner_area[2],
                           self.geometry.owner_area[1]:self.geometry.owner_area[3]] = True

        self.my_strawberry_mask = np.equal(
            self.geometry.type_map, self.geometry.types["strawberry"])
        self.my_strawberry_mask = np.logical_and(
            self.my_strawberry_mask, self.my_owner_mask)
        self.my_tomato_mask = np.equal(
            self.geometry.type_map, self.geometry.types["tomato"])
        self.my_tomato_mask = np.logical_and(
            self.my_tomato_mask, self.my_owner_mask)
        self.my_soil_mask = np.ones((self.width, self.height))
        self.my_soil_mask = np.logical_and(
            self.my_soil_mask, self.my_owner_mask)

    def field(self, name, model, savedir):
        """A field: the model replayed (model None) or run, with the values saved into savedir/precalc_<name>, 
        or not saved if savedir is None"""
        return PrecalculatedEnvironment(self.width, self.height, model, 
            None if savedir is None else pathlib.Path(savedir, f"precalc_{name}"))

    def create_epidemic(self, name, crop, seed, parameters):
        """An epidemic which spreads only on the cells planted with the crop"""
        immunity_mask = np.full([self.width, self.height], -2)
        immunity_mask[self.geometry.type_map == self.geometry.types[crop]] = 0
        infection_seeds = parameters["infection_seeds"]
        if infection_seeds is None:
            infection_seeds = default_infection_seeds(self.width)
        spread_dimension = parameters["spread_dimension"]
        if spread_dimension is None:
            spread_dimension = default_spread_dimension(self.width)
        return EpidemicSpreadEnvironment(name, self.width, self.height, seed, 
            p_transmission=parameters["p_transmission"], infection_duration=parameters["infection_duration"],
            spread_dimension=spread_dimension, infection_seeds=infection_seeds, immunity_mask=immunity_mask)

    def inner_proceed(self, delta_t=1.0):
        self.tylcv.proceed(delta_t)
        self.ccr.proceed(delta_t)
        self.soil.proceed(delta_t)

    def get_observation(self, pos):
        """For a given position and time creates a composite observation. The format for pos is [x,y,time]"""
        obs = {
            StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}

        tylcv = {
            StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}
        tylcv[StoredObservationIM.VALUE] = self.tylcv.get(pos[0], pos[1])
        obs["TYLCV"] = tylcv

        ccr = {
            StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}
        ccr[StoredObservationIM.VALUE] = self.ccr.get(pos[0], pos[1])
        obs["CCR"] = ccr

        soil = {
            StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], StoredObservationIM.TIME: pos[2]}
        soil[StoredObservationIM.VALUE] = self.soil.get(pos[0], pos[1])
        obs["Soil"] = soil

        return obs


    def animate_TO_DELETE(self, i):
        """Animates an environment by letting the environment proceed a timestep, then setting the values into image.
        FIXME: this does not belong here"""
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

    def animate_environment_TO_DELETE(self):
        """FIXME this does not belong here"""
        #fig, ax = plt.subplots()
        # im  = ax.imshow(env.value, cmap="gray", vmin=0, vmax=1.0)
        #axesimage  = ax.imshow(env.value, vmin=0, vmax=1)
        anim = animation.FuncAnimation(self.fig, partial(
            self.animate), frames=100, interval=50, blit=True)
        return anim


class WaterberryFarmInformationModel(StoredObservationIM):
    """The information model for the waterberry farm. It is basically a collection of three information models, one for each environmental model."""

    def __init__(self, width, height):
        """Creating the stuff"""
        super().__init__(width, height)
        self.im_tylcv = None
        self.im_ccr = None
        self.im_soil = None
        self.record = ObservationRecord(width, height)

    def add_observation(self, observation: dict):
        """It assumes that the observation is a dictionary with the components being the individual observations for TYLCV, CCR and soil humidity.
        This function records who observed which cell, and distributes the components of the observation to the subcomponents"""
        self.record.add(observation)
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

class WBF_IM_Composite(WaterberryFarmInformationModel):
    """WBF information model with a separately chosen scalar-field estimator for each of the three fields"""
    def __init__(self, width, height, im_tylcv, im_ccr, im_soil, name = "composite"):
        super().__init__(width, height)
        self.name = name
        self.im_tylcv, self.im_ccr, self.im_soil = im_tylcv, im_ccr, im_soil

class WBF_IM_DiskEstimator(WBF_IM_Composite):
    """WBF information model using a disk estimator with the specified disk radius for all three measures. For default value, we assume healthy for strawberry and tomato, and zero water for the humidity"""
    def __init__(self, width, height, disk_radius = None,
                 default_tylcv = 1.0, default_ccr = 1.0, default_soil = 0.0):
        super().__init__(width, height,
            DiskEstimateScalarFieldIM(width, height, disk_radius=disk_radius, default_value=default_tylcv),
            DiskEstimateScalarFieldIM(width, height, disk_radius=disk_radius, default_value=default_ccr),
            DiskEstimateScalarFieldIM(width, height, disk_radius=disk_radius, default_value=default_soil),
            name = "AD")

class WBF_IM_GaussianProcess(WBF_IM_Composite):
    """WBF information model using a gaussian process estimator for all three measures. For default value, we assume healthy for strawberry and tomato, and zero water for the humidity"""
    def __init__(self, width, height, gp_kernel = None, gp_restarts = 5, gp_normalize_y = False,
                 default_tylcv = 1.0, default_ccr = 1.0, default_soil = 0.0):
        super().__init__(width, height,
            GaussianProcessScalarFieldIM(width, height, gp_kernel, default_tylcv, gp_restarts, gp_normalize_y),
            GaussianProcessScalarFieldIM(width, height, gp_kernel, default_ccr, gp_restarts, gp_normalize_y),
            GaussianProcessScalarFieldIM(width, height, gp_kernel, default_soil, gp_restarts, gp_normalize_y),
            name = "GP")

class WBF_Score:
    """The ancestor of all the classes"""
    def score(self, env, im):
        return 0.0

    def __str__(self):
        return "WBF_Score: a scoring function that returns zero no matter what is being passed"

class WBF_Score_WeightedAsymmetric(WBF_Score):
    """An implementation of a score model which is using a weighted importance for each metric. For strawberry and tomato it is using an assymmetric metric which weights negative values differently compared to the positive ones"""
    def __init__(self, strawberry_importance = 0.2, # not lead to full crop loss
                strawberry_negative_importance = 10.0, # false negs 
                tomato_importance = 1.0, # leads to full crop loss
                tomato_negative_importance = 10.0, # false negs
                soil_importance = 0.1 # exact value less important
                ):
        self.strawberry_importance = strawberry_importance
        self.strawberry_negative_importance = strawberry_negative_importance
        self.tomato_importance = tomato_importance
        self.tomato_negative_importance = tomato_negative_importance
        self.soil_importance = soil_importance
    
    def __str__(self):
        return f"Importances: strawberry +{self.strawberry_importance}/-{self.strawberry_negative_importance} tomato +{self.tomato_importance}/-{self.tomato_negative_importance} soil {self.soil_importance}"

    def score(self, env, im):
        score = 0
        score += self.strawberry_importance * im_score_weighted_asymmetric(
            env.ccr, im.im_ccr, 1.0, self.strawberry_negative_importance, env.my_strawberry_mask)

        score += self.tomato_importance * im_score_weighted_asymmetric(
            env.tylcv, im.im_tylcv, 1.0, self.tomato_negative_importance, env.my_tomato_mask)

        score += self.soil_importance * \
            im_score_weighted(env.soil, im.im_soil, env.my_soil_mask)
        return score


class WBF_MultiScore(WBF_Score):
    """An implementation of a score model calculating a number of score components and recording them in a dictionary. Finally, it also calculates a custom score - which makes it equivalent to the 
    WBF_Score_WeightedAssymmetric, which it will replace
    """

    def __init__(self, strawberry_importance = 0.2, # not lead to full crop loss
                strawberry_negative_importance = 10.0, # false negs 
                tomato_importance = 1.0, # leads to full crop loss
                tomato_negative_importance = 10.0, # false negs
                soil_importance = 0.1, # exact value less important
                soil_negative_importance = 1
                ):
        self.strawberry_importance = strawberry_importance
        self.strawberry_negative_importance = strawberry_negative_importance
        self.tomato_importance = tomato_importance
        self.tomato_negative_importance = tomato_negative_importance
        self.soil_importance = soil_importance    
        self.soil_negative_importance = soil_negative_importance

    @staticmethod
    def score_components():
        """Returns the score components (eg for graphs)"""
        return ["strawberry-L1", "tomato-L1", "soil-L1", "strawberry-symmetric-L1", "tomato-symmetric-L1", "soil-symmetric-L1",
              "strawberry-asymmetric-L1", "tomato-asymmetric-L1", "soil-asymmetric-L1", "custom"]

    def score(self, env, im):
        """A more easily debuggable version of the score. Returns a dictionary with a number of components"""
        retval = {}
        retval["strawberry-L1"] = im_score(env.ccr, im.im_ccr)
        retval["tomato-L1"] = im_score(env.tylcv, im.im_tylcv)
        retval["soil-L1"] = im_score(env.soil, im.im_soil)

        retval["strawberry-symmetric-L1"] = im_score_weighted(env.ccr, im.im_ccr, env.my_strawberry_mask)
        retval["tomato-symmetric-L1"] = im_score_weighted(env.tylcv, im.im_tylcv, env.my_tomato_mask)
        retval["soil-symmetric-L1"] = im_score_weighted(env.soil, im.im_soil, env.my_soil_mask)

        # FIXME: maybe add here the assymmetry weights 
        retval["strawberry-asymmetric-L1"] =        im_score_weighted_asymmetric(
            env.ccr, im.im_ccr, 1.0, self.strawberry_negative_importance, env.my_strawberry_mask)
        retval["strawberry-asymmetric-L1-negimport"] = self.strawberry_negative_importance       
        
        retval["tomato-asymmetric-L1"] = im_score_weighted_asymmetric(env.tylcv, im.im_tylcv, 1.0, self.tomato_negative_importance, env.my_tomato_mask)
        retval["tomato-asymmetric-L1-negimport"] = self.tomato_negative_importance

        retval["soil-asymmetric-L1"] = im_score_weighted_asymmetric(env.soil, im.im_soil, 1.0, self.soil_negative_importance, env.my_soil_mask)
        retval["soil-assymmetric-L1-negimport"] = self.soil_negative_importance

        retval["custom"] = self.strawberry_importance * retval["strawberry-asymmetric-L1"] + self.tomato_importance * retval["tomato-asymmetric-L1"] + self.soil_importance * retval["soil-asymmetric-L1"]
        retval["custom-strawberry-importance"] = self.strawberry_importance
        retval["custom-tomato-importance"] = self.tomato_importance
        retval["custom-soil-importance"] = self.soil_importance

        # all weighted symmetric
        # all weighted assymmetric
        return retval


class WBF_Score_VoI(WBF_Score):
    """Value of information scores (absolute, expected, cost of ignorance, estimator-based) for the
    disease fields, together with their attribution to robots and timesteps (voi-credits).
    Stateful: an object scores exactly one run. See DESIGN-VOI.md"""

    VARIANTS = ["voi-absolute", "voi-expected", "voi-ignorance", "voi-estimator"]

    def __init__(self, v_pos = 100.0, # knowing it is infected: cost of damage
                 v_neg = 1.0, # knowing it is not infected: cost of verification
                 v_unknown = -3.0): # cost of ignorance
        self.v_pos = v_pos
        self.v_neg = v_neg
        self.v_unknown = v_unknown
        self.previous = {} # field name -> (expect, voi_est) at the previous scoring event
        self.last_time = -1 # the latest first-observation time already credited

    def __str__(self):
        return f"VoI: v_pos {self.v_pos} v_neg {self.v_neg} v_unknown {self.v_unknown}"

    @staticmethod
    def score_components():
        return WBF_Score_VoI.VARIANTS + ["voi"]

    def expect(self, value, probability = None):
        """Expected value of knowing the cells, from the probability that they are diseased if the 
        estimator provides it, otherwise derived from the estimated value"""
        p_pos = np.clip(2.0 * (1.0 - value), 0, 1) if probability is None else probability
        return p_pos * self.v_pos + (1 - p_pos) * self.v_neg

    def score(self, env, im):
        retval = {name: 0.0 for name in self.score_components()}
        credits = {} # (variant, robot, timestep) -> value
        xs, ys = im.record.indices(since=self.last_time + 1)
        new_cells = [(x, y) for x, y in zip(xs, ys)]
        observed = im.record.mask()
        for name, envfield, imfield, mask in [("TYLCV", env.tylcv, im.im_tylcv, env.my_tomato_mask),
                                              ("CCR", env.ccr, im.im_ccr, env.my_strawberry_mask)]:
            self.field_voi(name, envfield, imfield, mask, observed, new_cells, im.record, retval, credits)
        retval["voi"] = retval["voi-estimator"]
        retval["voi-credits"] = [{"variant": variant, "robot": robot, "timestep": timestep, "value": value}
                                 for (variant, robot, timestep), value in credits.items()]
        for x, y in new_cells:
            self.last_time = max(self.last_time, im.record.first(x, y)[1])
        return retval

    def field_voi(self, name, envfield, imfield, mask, observed, new_cells, record, retval, credits):
        """Adds the VoI totals of one disease field to retval, and its discovery credits and update terms
        since the previous scoring event to credits"""
        truth = np.where(envfield.value < 1.0, self.v_pos, self.v_neg)
        expect = self.expect(imfield.value, imfield.probability)
        c = imfield.confidence()
        voi_est = c * expect + (1 - c) * self.v_unknown
        unobserved = mask & ~observed
        absolute = np.sum(truth[mask & observed])
        retval["voi-absolute"] += absolute
        retval["voi-expected"] += absolute + np.sum(expect[unobserved])
        retval["voi-ignorance"] += absolute + self.v_unknown * np.sum(unobserved)
        retval["voi-estimator"] += np.sum(voi_est[mask])
        # the state before any observation: estimator default value, no confidence
        if name not in self.previous:
            self.previous[name] = (np.full(mask.shape, self.expect(imfield.default_value)),
                                   np.full(mask.shape, self.v_unknown))
        expect_prev, voi_est_prev = self.previous[name]
        # discovery credits of the cells first observed since the previous scoring event
        new_estimator = 0.0
        for x, y in new_cells:
            if not mask[x, y]:
                continue
            robot, timestep = record.first(x, y)
            for variant, value in [("voi-absolute", truth[x, y]),
                                   ("voi-ignorance", truth[x, y] - self.v_unknown),
                                   ("voi-expected", truth[x, y] - expect_prev[x, y]),
                                   ("voi-estimator", voi_est[x, y] - voi_est_prev[x, y])]:
                key = (variant, robot, timestep)
                credits[key] = credits.get(key, 0.0) + value
            new_estimator += voi_est[x, y] - voi_est_prev[x, y]
        # update terms: the changes not attributable to a robot
        updates = {"voi-absolute": 0.0, "voi-ignorance": 0.0,
                   "voi-expected": np.sum((expect - expect_prev)[unobserved]),
                   "voi-estimator": np.sum((voi_est - voi_est_prev)[mask]) - new_estimator}
        for variant, value in updates.items():
            key = (variant, None, None)
            credits[key] = credits.get(key, 0.0) + value
        self.previous[name] = (expect, voi_est)


def voi_credits(results, variant):
    """The VoI of the given variant added by each robot at each timestep. Returns a dict
    robot name -> np.array(timesteps), plus None -> the estimator update terms, placed at
    the timesteps of the scoring events."""
    timesteps = results["simulation-timestep"]
    credits = {name: np.zeros(timesteps) for name in results["robot-names"]}
    credits[None] = np.zeros(timesteps)
    for event in results["score-events"]:
        for row in event["score"]["voi-credits"]:
            if row["variant"] == variant:
                t = event["timestep"] if row["robot"] is None else row["timestep"]
                credits[row["robot"]][t] += row["value"]
    return credits



def get_datadir():
    """Returns the data directory associated with this project, ensuring that it exists.
    """
    datadir = pathlib.Path(Config()["WBF"]["data_dir"])
    datadir.mkdir(parents=True, exist_ok=True)
    return datadir


def create_wbfe(saved: bool, wbf_prec=None, typename="Miniberry-10"):
    """Helper function for the creation of a waterberry farm environment on which we can run experiments. It performs a caching process, if the files already exists, it just reloads them. This will save time for expensive simulations."""
    # p = pathlib.Path.cwd()
    # p = pathlib.Path(__file__).parent.resolve().parent.parent
    # datadir = pathlib.Path(p.parent, "__Temporary", p.name + "_data", typename)
    datadir = get_datadir()
    savedir = pathlib.Path(datadir, typename)
    savedir.mkdir(parents=True, exist_ok=True)
    path_geometry = pathlib.Path(savedir, "farm_geometry")
    path_environment = pathlib.Path(savedir, "farm_environment")
    if path_geometry.exists():
        logging.info("loading the geometry and environment from saved data")
        with bz2.open(path_geometry, "rb") as f:
            wbf = pickle.load(f)
        # with open(path_environment, "rb") as f:
        #    wbfe = pickle.load(f)
        #    wbfe.saved = saved
        logging.info("loading done")
        wbfe = WaterberryFarmEnvironment(wbf, saved, seed=10, savedir=savedir)
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
    wbfe = WaterberryFarmEnvironment(wbf, saved, seed=10, savedir=savedir)
    savedir.mkdir(parents=True, exist_ok=True)
    with bz2.open(path_geometry, "wb") as f:
        pickle.dump(wbf, f)
    with bz2.open(path_environment, "wb") as f:
        pickle.dump(wbfe, f)
    return wbf, wbfe, savedir
