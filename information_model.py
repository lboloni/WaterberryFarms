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


class AbstractScalarFieldIM(StoredObservationIM):
    """An abstract information model for scalar fields that keeps for each point the value and an uncertainty metric. A default value can be specified. The uncertainty metric is an estimate of the error at any given location. 
    """
    
    def __init__(self, width, height, default_value = 0):
        """Initializes the value to zero, the uncertainty to one. 
        FIXME: what exactly the uncertainty measures??? """
        super().__init__(width, height)
        self.default_value = default_value
        self.value = np.full((self.width, self.height), default_value)
        self.uncertainty = np.ones((self.width, self.height))

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
        _, uncertainty = self.estimate(self.observations)
        observations_new = self.observations.copy()
        observations_new.append(observation)
        _, uncertainty_new = self.estimate(observations_new)        
        return np.sum(np.abs(uncertainty - uncertainty_new))


class GaussianProcessScalarFieldIM(AbstractScalarFieldIM):
    """An information model for scalar fields where the estimation is happening
    using a GaussianProcess
    """

    def __init__(self, width, height, gp_kernel = None, default_value = 0.0):
        super().__init__(width, height, default_value)
        self.gp_kernel = gp_kernel

    def estimate(self, observations, prior_value, prior_uncertainty):
        # calculate the estimate for each gaussian process
        if prior_value != None or prior_uncertainty != None:
            Exception("GaussianProcessScalarFieldIM cannot handle priors")
        est = np.full([self.width,self.height], self.default_value)
        stdmap = np.zeros([self.width,self.height])
        if len(observations) == 0:
            return est, stdmap
        X = []
        Y = []
        for obs in observations:
            # Unclear if this rounding maters matters???
            X.append([round(obs[self.X]), round(obs[self.Y])])
            Y.append([obs[self.VALUE]])
        # fit the gaussian process
        kernel = RBF(length_scale = [2.0, 2.0], length_scale_bounds = [1, 10]) + WhiteKernel(noise_level=0.5)

        # rbf = RBF(length_scale = [2.0, 2.0], length_scale_bounds = "fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        gpr.fit(X,Y)
        x = []
        X = np.array(list(itertools.product(range(self.width), range(self.height))))
        Y, std = gpr.predict(X, return_std = True)
        for i, idx in enumerate(X):
            est[idx[0], idx[1]] = Y[i]
            stdmap[idx[0], idx[1]] = std[i]
        # print(std.sum())
        return est, stdmap

class PointEstimateScalarFieldIM(AbstractScalarFieldIM):
    """An information model which performs a point based estimation. In the precise point where we have an estimate, out uncertainty is zero, while everywhere else the uncertainty is 1.00
    """

    def __init__(self, width, height, default_value = 0.0):
        super().__init__(width, height, default_value)

    def estimate(self, observations, prior_value, prior_uncertainty):
        """Takes all the observations and estimates the value and the 
        uncertainty. This one processes all the observations, ignores the 
        timestamp, and assumes that each observation refers only to the current 
        point."""
        # value = np.ones((self.width, self.height)) * 0.5
        if prior_value != None:
            value = np.clone(prior_value)
        else:
            value = np.zeros((self.width, self.height)) 
        if prior_uncertainty != None:
            uncertainty = np.clone(prior_uncertainty)
        else:
            uncertainty = np.ones((self.width, self.height))
        for obs in observations:
            value[obs[self.X], obs[self.Y]] = obs[self.VALUE]
            uncertainty[obs[self.X], obs[self.Y]] = 0
        return value, uncertainty

class DiskEstimateScalarFieldIM(AbstractScalarFieldIM):
    """An information model which performs a disk based estimation.
    """

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
        self.maskdim = 2 * radius + 1
        self.mask = np.full((self.maskdim, self.maskdim), False, dtype=bool)
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if (math.sqrt((i*i + j*j)) <= radius):
                    self.mask[i + radius, j + radius] = True

        for obs in observations:
            #self.apply_value_iterate(value, obs[self.X], obs[self.Y], obs[self.VALUE], radius)
            self.apply_value_mask(value, uncertainty, obs[self.X], obs[self.Y], obs[self.VALUE])            
        return value, uncertainty

    def apply_value_mask(self, value, uncertainty, x, y, new_value):
        """Applies the value using a mask based approach"""
        # logging.info("apply_value_mask started")
        # create a true/false mask of the size of the environment for the application of the values. This is based on shifting the circular mask to the right location and resolving the situations where it overhangs the margins
        dimx = int(self.width)
        dimy = int(self.height)        
        mask2 = np.full((dimx, dimy), False, dtype=bool)
        maskp = self.mask[max(0, -x):min(self.maskdim, dimx-x), max(0, -y):min(self.maskdim,dimy-y)]
        mask2[max(0, x):min(dimx, x+self.maskdim), max(0, y):min(dimy,y+self.maskdim)] = maskp
        # and now we are assigning the new value for the part covered by the mask
        value[mask2] = new_value
        uncertainty[mask2] = 0.0
        # logging.info("apply_value_mask done")

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
    se = (env.value - im.value) ** 2
    weightedse = np.multiply(weightmap, se)
    weightedval= np.sqrt(np.sum(weightedse) / np.sum(weightmap)) 
    return -weightedval

def im_score_weighted(im, env, weightmap):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment. 
    The weightmap must be an array of the same size as the im value, and it must have its values between 0 (not interested) and 1 (interested)"""
    wm = weightmap / np.mean(weightmap)
    abserror = np.abs(env.value - im.value)
    weightederror = np.multiply(wm, abserror)
    return -np.mean(weightederror)

def im_score_weighted_asymmetric(im, env, weight_positive, weight_negative, weightmap):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment. 
    The weightmap must be an array of the same size as the im value, and it must have its values between 0 (not interested) and 1 (interested)
    Weights differently positive errors (when the im is larger than env) and negative errors (when env is larger than im)"""
    wm = weightmap / np.mean(weightmap)
    error_positive = np.multiply(wm * weight_positive, np.max(im.value - env.value, 0))
    error_negative = np.multiply(wm * weight_negative, np.max(env.value - im.value, 0))    
    return -np.mean(np.add(error_positive, error_negative))

