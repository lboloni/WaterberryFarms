import math
import itertools
import random
import logging
from functools import partial

import numpy as np
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# , WhiteKernel, RationalQuadratic, ExpSineSquared

import matplotlib.pyplot as plt
from matplotlib import animation, rc

from IPython.display import display, HTML


from Environment import Environment, PollutionModelEnvironment, EpidemicSpreadEnvironment

logging.basicConfig(level=logging.WARNING)

class InformationModel:
    """The ancestor of all information models. This defines the functions we can 
    call on these. They will need to be specialized to the different information models"""

    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
    
    def score(self, env: Environment):
        """Calculates a score that estimates the quality of this information model in modeling the
        specified environment"""
        return 0
    
    def add_observation(self, observation: dict):
        """Adds an observation as a dictionary with the fields value, x, y, timestamp etc. Different 
        implementations do different things with these observations (store, use it right away 
        to update the model etc.)"""
        pass
        
    def proceed(self, delta_t: float):
        """Proceed with the information model. The general assumption here is that after calling this
        the estimates are pre-computed and ready to be queried. Some implementations might model the
        evolution of the system as well."""
        pass

class ScalarFieldInformationModel_stored_observation(InformationModel):
    """An information model for scalar fields. It receives a series of observations. 
    It stores all the observations, and then uses them at estimate 
    time. This version is just keeping stored observations."""

    def __init__(self, name, width, height, estimation_type = "point", estimation_radius = 5):
        """Initializes the value to zero, the uncertainty to one. 
        FIXME: what exactly the uncertainty measures??? """
        super().__init__(name, width, height)
        self.observations = []
        self.estimation_type = estimation_type
        self.estimation_radius = estimation_radius 

    def score(self, env):
        """Scores the information model by finding the difference from the environment. In this case we are
        using models."""
        return np.sum(np.abs(env.value - self.value))

    def add_observation(self, observation):
        """The simplest way to add an observation is that we just record it."""
        self.observations.append(observation)

    def proceed(self, delta_t):
        """Proceeds a step in time. 
        At the current point, this basically performs an estimation, based on the information

        None of the currently used estimators 
        here """
        self.value, self.uncertainty = self.estimate(self.observations)

    def estimate(self, observations):
        """ Performs the estimate for every point in the environment. Returns a pair
        of the values and uncertainty as matrices for every point in the environment."""
        if self.estimation_type == "point":
            return self.estimate_with_point(observations)
        if self.estimation_type == "disk-auto":
            return self.estimate_with_disk(observations)
        if self.estimation_type == "disk-fixed":
            return self.estimate_with_disk(observations, self.estimation_radius)
        if self.estimation_type == "gaussian-process":
            return self.estimate_with_gaussian_process(observations)
        raise Exception(f"Unknown estimation type {self.estimation_type}")


    def estimate_with_point(self, observations):
        """Takes all the observations and estimates the value and the uncertainty. This one processes all the 
        observations, ignores the timestamp, and assumes that each observation refers only to the current point."""
        # value = np.ones((self.width, self.height)) * 0.5
        value = np.zeros((self.width, self.height)) 
        uncertainty = np.ones((self.width, self.height))
        for obs in observations:
            value[obs["x"], obs["y"]] = obs["value"]
            uncertainty[obs["x"], obs["y"]] = 0
        return value, uncertainty

    def estimate_with_disk(self, observations, radius=None):
        """Consider that we are estimating them with a disk of a certain radius r. We set the values to it.
        The radius r can be dynamically calculated such that the total disks achieve 2x the coverage of the 
        area. sqrt((height * width * 2) / pi). 
        Then we set the values in a disk"""
        value = np.zeros((self.width, self.height)) 
        uncertainty = np.ones((self.width, self.height))
        if radius == None:
            radius = 1+math.sqrt((self.height * self.width * 2) / (math.pi * len(observations)))
        # FIXME: this is not very efficient, it can be made more efficient by iterating just one corner
        # and finding values.
        for obs in observations:
            x = obs["x"]
            y = obs["y"]
            for i in range(round(max(0, x-radius)), round(min(x+radius, self.width))):
                for j in range(round(max(0, y-radius)), round(min(y+radius, self.height))):
                    if (math.sqrt((i-x)*(i-x) + (j-y)*(j-y)) <= radius):
                        # FIXME: could do better... with averaging taking into consideration the i and j
                        value[i, j] = obs["value"]
                        uncertainty[i, j] = 0
        return value, uncertainty

    def estimate_with_gaussian_process(self, observations):
        # calculate the estimate for each gaussian process
        est = np.zeros([self.width,self.height])
        stdmap = np.zeros([self.width,self.height])
        if len(observations) == 0:
            return est, stdmap
        X = []
        Y = []
        for obs in observations:
            # Unclear if this rounding maters matters???
            X.append([round(obs["x"]), round(obs["y"])])
            Y.append([obs["value"]])
        # fit the gaussian process
        #rbf = 2.0 * RBF(length_scale = [1.0, 1.0], length_scale_bounds = [1, 10])
        rbf = RBF(length_scale = [2.0, 2.0], length_scale_bounds = "fixed")
        gpr = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=5)
        gpr.fit(X,Y)
        x = []
        X = np.array(list(itertools.product(range(self.width), range(self.height))))
        Y, std = gpr.predict(X, return_std = True)
        for i, idx in enumerate(X):
            est[idx[0], idx[1]] = Y[i]
            stdmap[idx[0], idx[1]] = std[i]
        # print(std.sum())
        return est, stdmap

    def estimate_voi(self, observation):
        """The voi of the observation is the reduction of the uncertainty.
        FIXME this can be made different for the GP"""
        _, uncertainty = self.estimate(self.observations)
        observations_new = observations.clone()
        observations_new.append(observation)
        _, uncertainty_new = self.estimate(observations_new)        
        return np.sum(np.abs(uncertainty - uncertainty_new))

