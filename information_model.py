"""
information_model.py

Implementation of the basic InformationModel class and several practical information models: Gaussian Process and Adaptive Disk
"""

import math
import itertools
import random
import logging
from functools import partial

import numpy as np
from scipy import signal
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib import animation, rc
# import unittest
import timeit

from environment import Environment, DissipationModelEnvironment, EpidemicSpreadEnvironment

logging.basicConfig(level=logging.WARNING)

class InformationModel:
    """The ancestor of all information models. This defines the functions we can 
    call on these. They will need to be specialized to the different information models"""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.name = "GenericInformationModel"
    
    def add_observation(self, observation: dict):
        """Adds an observation as a dictionary with the fields value, x, y, 
        timestamp etc. Different implementations do different things with these 
        observations (store, use it right away to update the model etc.)"""
        pass
        
    def proceed(self, delta_t: float):
        """Proceed with the information model. The general assumption here is 
        that after calling this the estimates are pre-computed and ready to be 
        queried. Some implementations might model the evolution of the system 
        as well."""
        pass


class StoredObservationIM(InformationModel):
    """An information model that receives a series of 
    observations. Observations are dictionaries with the specific values as specified in the constants below."""

    VALUE = "value"
    X = "x"
    Y = "y"
    LOCATION = "location"
    TIME = "time"
    CONFIDENCE = "confidence"
    RANGE = "range"

    def __init__(self, width, height):
        super().__init__(width, height)
        self.observations = []

    def add_observation(self, observation):
        """The simplest way to add an observation is that we just record it."""
        self.observations.append(observation)


class ObservationRecord:
    """Records for every observed cell the number of observations and the first observation (robot, time).
    Sparse: only observed cells are stored. When several robots observe a new cell in the same timestep,
    the first one added (first in robot name order in simulate_1day) is credited."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cells = {}  # (x, y) -> {"count": n, "first-robot": name, "first-time": t}

    def add(self, observation):
        cell = (observation["x"], observation["y"])
        if cell in self.cells:
            self.cells[cell]["count"] += 1
        else:
            self.cells[cell] = {"count": 1, "first-robot": observation["robot"],
                                "first-time": observation["time"]}

    def count(self, x, y):
        """Number of observations of the cell, 0 if never observed"""
        return self.cells[(x, y)]["count"] if (x, y) in self.cells else 0

    def first(self, x, y):
        """(robot, time) of the first observation of the cell, None if never observed"""
        if (x, y) not in self.cells:
            return None
        cell = self.cells[(x, y)]
        return cell["first-robot"], cell["first-time"]

    def indices(self, robot = None, since = None, min_count = 1):
        """Index arrays (xs, ys) of the observed cells, usable as array[xs, ys].
        robot: only cells first observed by this robot
        since: only cells first observed at time >= since
        min_count: only cells observed at least this many times (2 = repeated)"""
        selected = [cell for cell, rec in self.cells.items()
                    if (robot is None or rec["first-robot"] == robot)
                    and (since is None or rec["first-time"] >= since)
                    and rec["count"] >= min_count]
        xs = np.array([cell[0] for cell in selected], dtype=int)
        ys = np.array([cell[1] for cell in selected], dtype=int)
        return xs, ys

    def mask(self, **filters):
        """Dense boolean (width x height) array of the cells selected by indices(**filters)"""
        m = np.zeros((self.width, self.height), dtype=bool)
        m[self.indices(**filters)] = True
        return m

    def first_counts(self):
        """Dict robot name -> number of cells that robot observed first"""
        counts = {}
        for rec in self.cells.values():
            counts[rec["first-robot"]] = counts.get(rec["first-robot"], 0) + 1
        return counts


DISEASED = 0.75 # the values below this are diseased: infected (0.5) or destroyed (0.0)


def latest_per_cell(observations):
    """The latest observed value of every observed cell, as (cells: n x 2 int array, values: n array)"""
    latest = {}
    for obs in observations:
        latest[(int(obs["x"]), int(obs["y"]))] = obs["value"]
    cells = np.array(list(latest.keys()), dtype=int).reshape(-1, 2)
    return cells, np.array(list(latest.values()), dtype=float)


def grid_cells(width, height):
    """The cells of the grid as a (width * height) x 2 array, in the order of reshape(width, height)"""
    return np.array(list(itertools.product(range(width), range(height))))


def distance_uncertainty(distance, length_scale):
    """The uncertainty of the "distance" kind: 0 at an observation, approaching 1 far from the observations"""
    return 1.0 - np.exp(-distance / length_scale)


def probability_uncertainty(probability):
    """The uncertainty of the "probability" kind: 0 when certain, 1 when the probability is 0.5"""
    return 2.0 * np.sqrt(probability * (1.0 - probability))


def value_from_probability(probability, observations):
    """The expected value of a disease field given the probability that a cell is diseased: healthy is 1.0, 
    diseased the mean of the observed diseased values (0.5 if none). Observed cells keep their observed value."""
    cells, values = latest_per_cell(observations)
    diseased = values[values < DISEASED]
    diseased_value = diseased.mean() if len(diseased) > 0 else 0.5
    value = 1.0 - probability * (1.0 - diseased_value)
    value[cells[:, 0], cells[:, 1]] = values
    return value


class AbstractScalarFieldIM(StoredObservationIM):
    """An abstract information model for scalar fields that keeps for each point the value and an uncertainty metric. A default value can be specified. 
    UNCERTAINTY states what the uncertainty array means:
        coverage    - 0 where an observation covers the cell, 1 elsewhere
        std         - a posterior standard deviation, in value units (prior_std without observations)
        probability - 2 sqrt(p (1-p)) of the probability p that the cell is diseased
        distance    - 1 - exp(-d / length_scale) of the distance d to the nearest observation
        none        - 1 everywhere
    Disease estimators may also provide probability: the probability that the cell is diseased (value < 0.75).
    """
    UNCERTAINTY = "none"
    
    def __init__(self, width, height, default_value = 0):
        """Initializes the value to the default value, the uncertainty to one."""
        super().__init__(width, height)
        self.default_value = default_value
        self.value = np.full((self.width, self.height), default_value)
        self.uncertainty = np.ones((self.width, self.height))
        self.probability = None
        self.prior_std = 1.0

    def confidence(self):
        """The confidence in the estimate of every cell, in [0, 1] (1 = certain), comparable across estimators"""
        if self.UNCERTAINTY == "std":
            return np.clip(1.0 - self.uncertainty / self.prior_std, 0, 1)
        return np.clip(1.0 - self.uncertainty, 0, 1)

    def proceed(self, delta_t):
        """Proceeds a step in time. At the current point, this basically performs an estimation, based on the observations. 
        None of the currently used estimators have the ability to make predictions, but this might be the case later on"""
        self.value, self.uncertainty = self.estimate(self.observations, None, None)

    def estimate(self, observations, prior_value: None, prior_uncertainty: None):
        """Performs the estimate for every point in the environment. Returns a the posterior values and uncertainty as arrays for every point in the environment. 
        The observations are the ones that have not been integrated into the prior"""
        raise Exception(f"Trying to call estimate in the abstract class.")

    def estimate_voi(self, observation):
        """The voi of the observation is the reduction of the uncertainty.
        FIXME this can be made different for the GP"""
        _, uncertainty = self.estimate(self.observations, None, None)
        observations_new = self.observations.copy()
        observations_new.append(observation)
        _, uncertainty_new = self.estimate(observations_new, None, None)
        return np.sum(np.abs(uncertainty - uncertainty_new))


class GaussianProcessScalarFieldIM(AbstractScalarFieldIM):
    """An information model for scalar fields where the estimation is happening
    using a GaussianProcess (with normalize_y, this is ordinary kriging). The uncertainty is the posterior 
    standard deviation; prior_std is the standard deviation of the fitted prior, used by confidence().
    max_observations: if not None, the observations are reduced to the latest value of every cell, and 
    at most max_observations of these, chosen evenly in the order of observation
    """
    UNCERTAINTY = "std"

    def __init__(self, width, height, gp_kernel = None, default_value = 0.0, n_restarts_optimizer = 5, normalize_y = False, max_observations = None):
        super().__init__(width, height, default_value)
        self.gp_kernel = gp_kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.max_observations = max_observations

    def training_data(self, observations):
        """The training inputs (cells) and targets of the GP"""
        if self.max_observations is None:
            # Unclear if this rounding maters matters???
            X = [[round(obs[self.X]), round(obs[self.Y])] for obs in observations]
            Y = [[obs[self.VALUE]] for obs in observations]
            return X, Y
        cells, values = latest_per_cell(observations)
        keep = np.unique(np.linspace(0, len(values) - 1, min(len(values), self.max_observations)).astype(int))
        return cells[keep].tolist(), values[keep].reshape(-1, 1).tolist()

    def fit_predict(self, X, Y, points):
        """Fits the GP to the training data and predicts the mean and std at the points; sets prior_std"""
        kernel = self.gp_kernel
        if kernel is None:
            kernel = RBF(length_scale = [2.0, 2.0], length_scale_bounds = [1, 10]) + WhiteKernel(noise_level=0.5)
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self.n_restarts_optimizer, normalize_y=self.normalize_y, random_state=0)
        gpr.fit(X, Y)
        mean, std = gpr.predict(points, return_std = True)
        scale = float(np.ravel(gpr._y_train_std)[0]) if self.normalize_y else 1.0
        self.prior_std = float(np.sqrt(gpr.kernel_.diag(np.zeros((1, 2)))[0])) * scale
        return np.ravel(mean), np.ravel(std)

    def estimate(self, observations, prior_value, prior_uncertainty):
        # calculate the estimate for each gaussian process
        if prior_value is not None or prior_uncertainty is not None:
            raise Exception("GaussianProcessScalarFieldIM cannot handle priors")
        est = np.full([self.width,self.height], self.default_value)
        stdmap = np.ones([self.width,self.height])
        if len(observations) == 0:
            self.prior_std = 1.0
            return est, stdmap
        X, Y = self.training_data(observations)
        points = grid_cells(self.width, self.height)
        Y, std = self.fit_predict(X, Y, points)
        est[points[:, 0], points[:, 1]] = Y
        stdmap[points[:, 0], points[:, 1]] = std
        return est, stdmap


class LocalGPScalarFieldIM(GaussianProcessScalarFieldIM):
    """Local Gaussian processes: the grid is divided into tiles of tile_size, and every tile has its own GP, 
    trained on the observations within the tile extended by overlap. Tiles without observations keep the 
    default value and the prior. The cost grows with the observations per tile, not with all observations."""

    def __init__(self, width, height, gp_kernel = None, default_value = 0.0, n_restarts_optimizer = 5, normalize_y = False, max_observations = None, tile_size = 20, overlap = 5):
        super().__init__(width, height, gp_kernel, default_value, n_restarts_optimizer, normalize_y, max_observations)
        self.tile_size = tile_size
        self.overlap = overlap

    def estimate(self, observations, prior_value, prior_uncertainty):
        est = np.full([self.width,self.height], self.default_value, dtype=float)
        stdmap = np.ones([self.width,self.height])
        X, Y = self.training_data(observations) if observations else ([], [])
        X, Y = np.array(X).reshape(-1, 2), np.array(Y).reshape(-1, 1)
        prior_stds = []
        for x0 in range(0, self.width, self.tile_size):
            for y0 in range(0, self.height, self.tile_size):
                x1, y1 = min(x0 + self.tile_size, self.width), min(y0 + self.tile_size, self.height)
                inside = ((X[:, 0] >= x0 - self.overlap) & (X[:, 0] < x1 + self.overlap) &
                          (X[:, 1] >= y0 - self.overlap) & (X[:, 1] < y1 + self.overlap))
                if not inside.any():
                    continue
                points = np.array(list(itertools.product(range(x0, x1), range(y0, y1))))
                mean, std = self.fit_predict(X[inside], Y[inside], points)
                prior_stds.append(self.prior_std)
                est[points[:, 0], points[:, 1]] = mean
                stdmap[points[:, 0], points[:, 1]] = std
        self.prior_std = max(prior_stds) if prior_stds else 1.0
        return est, stdmap


class IndicatorGPScalarFieldIM(GaussianProcessScalarFieldIM):
    """Indicator kriging for a disease field: a GP regression of the indicator of the diseased cells 
    (value < 0.75), whose clipped mean is the probability that a cell is diseased. The prior mean of the 
    indicator is 0, i.e. healthy."""
    UNCERTAINTY = "probability"

    def estimate(self, observations, prior_value, prior_uncertainty):
        indicators = [dict(obs, value=1.0 if obs[self.VALUE] < DISEASED else 0.0) for obs in observations]
        if indicators:
            X, Y = self.training_data(indicators)
            mean, _ = self.fit_predict(X, Y, grid_cells(self.width, self.height))
            probability = np.clip(mean, 0, 1).reshape(self.width, self.height)
        else:
            probability = np.zeros((self.width, self.height))
        self.probability = probability
        return value_from_probability(probability, observations), probability_uncertainty(probability)


class NearestScalarFieldIM(AbstractScalarFieldIM):
    """Nearest-neighbor (Voronoi) interpolation: every cell takes the latest value of the nearest observed cell"""
    UNCERTAINTY = "distance"

    def __init__(self, width, height, default_value = 0.0, length_scale = 3.0):
        super().__init__(width, height, default_value)
        self.length_scale = length_scale

    def estimate(self, observations, prior_value, prior_uncertainty):
        if not observations:
            return np.full((self.width, self.height), self.default_value, dtype=float), np.ones((self.width, self.height))
        cells, values = latest_per_cell(observations)
        distance, index = cKDTree(cells).query(grid_cells(self.width, self.height))
        return (values[index].reshape(self.width, self.height),
                distance_uncertainty(distance, self.length_scale).reshape(self.width, self.height))


class IDWScalarFieldIM(AbstractScalarFieldIM):
    """Inverse distance weighting: every cell is the mean of the k nearest observed cells, weighted by 
    1 / distance^power; observed cells keep their value"""
    UNCERTAINTY = "distance"

    def __init__(self, width, height, default_value = 0.0, power = 2.0, k = 8, length_scale = 3.0):
        super().__init__(width, height, default_value)
        self.power, self.k, self.length_scale = power, k, length_scale

    def estimate(self, observations, prior_value, prior_uncertainty):
        if not observations:
            return np.full((self.width, self.height), self.default_value, dtype=float), np.ones((self.width, self.height))
        cells, values = latest_per_cell(observations)
        grid = grid_cells(self.width, self.height)
        k = min(self.k, len(values))
        distance, index = cKDTree(cells).query(grid, k=k)
        distance, index = distance.reshape(len(grid), k), index.reshape(len(grid), k)
        weights = 1.0 / np.maximum(distance, 1e-12) ** self.power
        value = (weights * values[index]).sum(axis=1) / weights.sum(axis=1)
        exact = distance[:, 0] == 0
        value[exact] = values[index[exact, 0]]
        return (value.reshape(self.width, self.height),
                distance_uncertainty(distance[:, 0], self.length_scale).reshape(self.width, self.height))


class RBFScalarFieldIM(AbstractScalarFieldIM):
    """Radial basis function interpolation (scipy RBFInterpolator, with a constant polynomial term, 
    so that collinear observations along a trajectory are supported), clipped to [0, 1]. 
    neighbors: if not None, every cell is interpolated from its nearest observed cells only."""
    UNCERTAINTY = "distance"

    def __init__(self, width, height, default_value = 0.0, kernel = "linear", epsilon = 1.0, smoothing = 0.0, neighbors = 32, length_scale = 3.0):
        super().__init__(width, height, default_value)
        self.kernel, self.epsilon, self.smoothing, self.neighbors, self.length_scale = kernel, epsilon, smoothing, neighbors, length_scale

    def estimate(self, observations, prior_value, prior_uncertainty):
        if not observations:
            return np.full((self.width, self.height), self.default_value, dtype=float), np.ones((self.width, self.height))
        cells, values = latest_per_cell(observations)
        grid = grid_cells(self.width, self.height)
        neighbors = None if self.neighbors is None else min(self.neighbors, len(values))
        interpolator = RBFInterpolator(cells, values, kernel=self.kernel, epsilon=self.epsilon, 
                                       smoothing=self.smoothing, neighbors=neighbors, degree=0)
        value = np.clip(interpolator(grid), 0, 1)
        distance, _ = cKDTree(cells).query(grid)
        return (value.reshape(self.width, self.height),
                distance_uncertainty(distance, self.length_scale).reshape(self.width, self.height))


class OccupancyGridIM(AbstractScalarFieldIM):
    """A Bayesian occupancy grid of the diseased cells (value < 0.75) of a disease field, in log-odds. 
    The latest observation of every cell is evidence for that cell and, weighted by exp(-d^2 / footprint^2), 
    for the cells around it; the sensor model is P(observed diseased | diseased) = p_hit and 
    P(observed diseased | healthy) = p_false_alarm. Every cell counts once, as the simulated sensor is 
    deterministic (a parked robot gives no new evidence)."""
    UNCERTAINTY = "probability"

    def __init__(self, width, height, default_value = 1.0, p_hit = 0.95, p_false_alarm = 0.05, footprint = 2.0, prior = 0.1):
        super().__init__(width, height, default_value)
        self.p_hit, self.p_false_alarm, self.footprint, self.prior = p_hit, p_false_alarm, footprint, prior

    def estimate(self, observations, prior_value, prior_uncertainty):
        logodds = np.full((self.width, self.height), math.log(self.prior / (1 - self.prior)))
        cells, values = latest_per_cell(observations)
        radius = int(math.ceil(3 * self.footprint))
        for (x, y), observed in zip(cells, values):
            if observed < DISEASED:
                evidence = math.log(self.p_hit / self.p_false_alarm)
            else:
                evidence = math.log((1 - self.p_hit) / (1 - self.p_false_alarm))
            x0, x1 = max(0, x - radius), min(self.width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(self.height, y + radius + 1)
            dx, dy = np.meshgrid(np.arange(x0, x1) - x, np.arange(y0, y1) - y, indexing="ij")
            weight = np.exp(-(dx**2 + dy**2) / self.footprint**2) if self.footprint > 0 else (dx**2 + dy**2 == 0) * 1.0
            logodds[x0:x1, y0:y1] += weight * evidence
        probability = 1.0 / (1.0 + np.exp(-np.clip(logodds, -30, 30)))
        self.probability = probability
        return value_from_probability(probability, observations), probability_uncertainty(probability)


class MRFScalarFieldIM(AbstractScalarFieldIM):
    """An Ising Markov random field of the diseased cells of a disease field: spins +1 (diseased) / -1 
    (healthy), coupled to their 4 neighbors with coupling, biased by the prior probability; the observed 
    cells are clamped. The marginal probabilities are approximated by mean-field iterations."""
    UNCERTAINTY = "probability"

    def __init__(self, width, height, default_value = 1.0, coupling = 0.5, prior = 0.1, iterations = 50):
        super().__init__(width, height, default_value)
        self.coupling, self.prior, self.iterations = coupling, prior, iterations

    def estimate(self, observations, prior_value, prior_uncertainty):
        bias = 0.5 * math.log(self.prior / (1 - self.prior))
        magnetization = np.full((self.width, self.height), 2 * self.prior - 1)
        cells, values = latest_per_cell(observations)
        spins = np.where(values < DISEASED, 1.0, -1.0)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for _ in range(self.iterations):
            magnetization[cells[:, 0], cells[:, 1]] = spins
            neighbors = signal.convolve2d(magnetization, kernel, mode="same")
            magnetization = np.tanh(bias + self.coupling * neighbors)
        magnetization[cells[:, 0], cells[:, 1]] = spins
        probability = (1 + magnetization) / 2
        self.probability = probability
        return value_from_probability(probability, observations), probability_uncertainty(probability)

class PointEstimateScalarFieldIM(AbstractScalarFieldIM):
    """An information model which performs a point based estimation. In the precise point where we have an estimate, out uncertainty is zero, while everywhere else the uncertainty is 1.00
    """
    UNCERTAINTY = "coverage"

    def __init__(self, width, height, default_value = 0.0):
        super().__init__(width, height, default_value)

    def estimate(self, observations, prior_value, prior_uncertainty):
        """Takes all the observations and estimates the value and the 
        uncertainty. This one processes all the observations, ignores the 
        timestamp, and assumes that each observation refers only to the current 
        point."""
        # value = np.ones((self.width, self.height)) * 0.5
        if prior_value is not None:
            value = np.copy(prior_value)
        else:
            value = np.full((self.width, self.height), self.default_value)
        if prior_uncertainty is not None:
            uncertainty = np.copy(prior_uncertainty)
        else:
            uncertainty = np.ones((self.width, self.height))
        for obs in observations:
            value[obs[self.X], obs[self.Y]] = obs[self.VALUE]
            uncertainty[obs[self.X], obs[self.Y]] = 0
        return value, uncertainty

class DiskEstimateScalarFieldIM(AbstractScalarFieldIM):
    """An information model which performs a disk based estimation.
    """
    UNCERTAINTY = "coverage"

    def __init__(self, width, height, disk_radius=5, default_value=0):
        super().__init__(width, height, default_value)
        self.disk_radius = disk_radius
        self.mask = None

    def estimate(self, observations, prior_value, prior_uncertainty, radius=None):
        """Consider that we are estimating them with a disk of a certain radius 
        r. The radius r can be dynamically calculated such that the total disks achieve 2x the coverage of the area. sqrt((height * width * 2) / pi). 
        Later disks overwrite earlier disks.
        FIXME: Areas that have no coverage have uncertainty 1, while areas that fit into a disk have an uncertainty 0."""
        value = np.full((self.width, self.height), self.default_value, dtype=np.float64) 
        uncertainty = np.ones((self.width, self.height), dtype=np.float64)
        if self.disk_radius == None:
            if len(observations) == 0:
                radius = 1
            else:
                radius = int(1+math.sqrt((self.height * self.width * 2) / (math.pi * len(observations))))
        else:
            radius = self.disk_radius

        # create a mask array
        self.mask_radius = radius
        self.maskdim = 2 * radius + 1
        self.mask = np.full((self.maskdim, self.maskdim), False, dtype=bool)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if (math.sqrt((i*i + j*j)) <= radius):
                    self.mask[i + radius, j + radius] = True

        for obs in observations:
            #self.apply_value_iterate(value, obs[self.X], obs[self.Y], obs[self.VALUE], radius)
            self.apply_value_mask(value, uncertainty, obs[self.X], obs[self.Y], obs[self.VALUE])            
        return value, uncertainty

    def apply_value_mask(self, value, uncertainty, x, y, new_value):
        """Applies the value using a mask based approach"""
        x = int(x)
        y = int(y)
        radius = self.mask_radius
        value_x_start = max(0, x - radius)
        value_x_end = min(self.width, x + radius + 1)
        value_y_start = max(0, y - radius)
        value_y_end = min(self.height, y + radius + 1)

        mask_x_start = value_x_start - (x - radius)
        mask_x_end = mask_x_start + value_x_end - value_x_start
        mask_y_start = value_y_start - (y - radius)
        mask_y_end = mask_y_start + value_y_end - value_y_start

        mask = self.mask[mask_x_start:mask_x_end, mask_y_start:mask_y_end]
        value_slice = value[value_x_start:value_x_end, value_y_start:value_y_end]
        uncertainty_slice = uncertainty[value_x_start:value_x_end, value_y_start:value_y_end]
        value_slice[mask] = new_value
        uncertainty_slice[mask] = 0.0

def im_score(im, env):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment."""
    return -np.mean(np.abs(env.value - im.value))

def im_score_rmse_scikit(im, env):
    """Scores the information model by finding the RMSE between the information model values and the real values in the environment
    FIXME: I think that this is not working very well for 2D areas. On the other hand, it has built-in weights
    """
    return -mean_squared_error(env.value, im.value, squared=False)

def im_score_rmse(im, env):
    """Scores the information model by finding the RMSE between the information model values and the real values in the environment"""
    se = (env.value - im.value) ** 2
    return -np.sqrt(np.mean(se))

def im_score_rmse_weighted(im, env, weightmap):
    """An empty area of interest (e.g. a crop that is not planted) has no error."""
    if np.sum(weightmap) == 0:
        return 0.0
    se = (env.value - im.value) ** 2
    weightedse = np.multiply(weightmap, se)
    weightedval= np.sqrt(np.sum(weightedse) / np.sum(weightmap)) 
    return -weightedval

def im_score_weighted(im, env, weightmap):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment. 
    The weightmap must be an array of the same size as the im value, and it must have its values between 0 (not interested) and 1 (interested).
    An empty area of interest (e.g. a crop that is not planted) has no error."""
    if np.sum(weightmap) == 0:
        return 0.0
    wm = weightmap / np.mean(weightmap)
    abserror = np.abs(env.value - im.value)
    weightederror = np.multiply(wm, abserror)
    return -np.mean(weightederror)

def im_score_weighted_asymmetric(im, env, weight_positive, weight_negative, weightmap):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment. 
    The weightmap must be an array of the same size as the im value, and it must have its values between 0 (not interested) and 1 (interested)
    Weights differently positive errors (when the im is larger than env) and negative errors (when env is larger than im).
    An empty area of interest (e.g. a crop that is not planted) has no error."""
    if np.sum(weightmap) == 0:
        return 0.0
    wm = weightmap / np.mean(weightmap)
    error_positive = np.multiply(wm * weight_positive, np.maximum(im.value - env.value, 0))
    error_negative = np.multiply(wm * weight_negative, np.maximum(env.value - im.value, 0))
    return -np.mean(np.add(error_positive, error_negative))

