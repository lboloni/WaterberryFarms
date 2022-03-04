import math
import itertools
import random
import logging
from functools import partial

import numpy as np
from scipy import signal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import matplotlib.pyplot as plt
from matplotlib import animation, rc
import unittest
import timeit

# from IPython.display import display, HTML

from Environment import Environment, DissipationModelEnvironment, EpidemicSpreadEnvironment

logging.basicConfig(level=logging.WARNING)

class InformationModel:
    """The ancestor of all information models. This defines the functions we can 
    call on these. They will need to be specialized to the different information models"""

    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
    
    def score(self, env: Environment):
        """Calculates a score that estimates the quality of this information 
        model in modeling the specified environment"""
        return 0
    
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

    def __init__(self, name, width, height):
        super().__init__(name, width, height)
        self.observations = []

    def add_observation(self, observation):
        """The simplest way to add an observation is that we just record it."""
        self.observations.append(observation)


class AbstractScalarFieldIM(StoredObservationIM):
    """An abstract information model for scalar fields that keeps for each point the value and an uncertainty metric. The uncertainty metric is an estimate of the error at any given location. 
    """
    
    def __init__(self, name, width, height, estimation_type = "point", \
                 estimation_radius = 5, gp_kernel = None):
        """Initializes the value to zero, the uncertainty to one. 
        FIXME: what exactly the uncertainty measures??? """
        super().__init__(name, width, height)
        self.value = np.zeros((self.width, self.height))
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

def im_score(im, env):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment."""
    return np.mean(np.abs(env.value - im.value))

def im_score_weighted(im, env, weightmap):
    """Scores the information model by finding the average absolute difference between the prediction of the information model and the real values in the environment."""
    wm = weightmap / np.mean(weightmap)
    abserror = np.abs(env.value - im.value)
    weightederror = np.multiply(wm, abserror)
    return np.mean(weightederror)


class GaussianProcessScalarFieldIM(AbstractScalarFieldIM):
    """An information model for scalar fields where the estimation is happening
    using a GaussianProcess
    """

    def __init__(self, name, width, height, gp_kernel = None):
        super().__init__(name, width, height)
        self.gp_kernel = gp_kernel

    def estimate(self, observations, prior_value, prior_uncertainty):
        # calculate the estimate for each gaussian process
        if prior_value != None or prior_uncertainty != None:
            Exception("GaussianProcessScalarFieldIM cannot handle priors")
        est = np.zeros([self.width,self.height])
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

    def __init__(self, name, width, height):
        super().__init__(name, width, height)

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

    def __init__(self, name, width, height, disk_radius=5):
        super().__init__(name, width, height)
        self.disk_radius = disk_radius
        self.mask = None

    def estimate(self, observations, prior_value, prior_uncertainty, radius=None):
        """Consider that we are estimating them with a disk of a certain radius 
        r. The radius r can be dynamically calculated such that the total disks achieve 2x the coverage of the area. sqrt((height * width * 2) / pi). 
        Later disks overwrite earlier disks.
        FIXME: Areas that have no coverage have uncertainty 1, while areas that fit into a disk have an uncertainty 0."""
        value = np.zeros((self.width, self.height)) 
        uncertainty = np.ones((self.width, self.height))
        if self.disk_radius == None:
            radius = int(1+math.sqrt((self.height * self.width * 2) / (math.pi * len(observations))))
        else:
            radius = self.disk_radius
        # FIXME: this is not very efficient, it can be made more efficient by iterating just one corner
        # and finding values.
        # FIXME: this could be made more efficient by creating a circular mask and then moving it... 

        # create a mask array
        maskdim = 2 * radius + 1
        self.mask = np.full((maskdim, maskdim), False, dtype=bool)
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if (math.sqrt((i*i + j*j)) <= radius):
                    self.mask[i + radius, j + radius] = True

        for obs in observations:
            #self.apply_value_iterate(value, obs[self.X], obs[self.Y], obs[self.VALUE], radius)
            self.apply_value_mask(value, obs[self.X], obs[self.Y], obs[self.VALUE], radius)            
        return value, uncertainty


    def apply_value_iterate(self, value, x, y, new_value, radius):
        """Applies the value in a radius by iteration"""
        for i in range(round(max(0, x-radius)), round(min(x+radius, self.width))):
            for j in range(round(max(0, y-radius)), round(min(y+radius, self.height))):
                if (math.sqrt((i-x)*(i-x) + (j-y)*(j-y)) <= radius):
                    # FIXME: could do better... with averaging taking into consideration the i and j
                    value[i, j] = new_value
                    # uncertainty[i, j] = 0

    def apply_value_mask(self, value, x, y, new_value, radius):
        """Applies the value using a mask based approach"""
        # logging.info("apply_value_mask started")
        maskdim = 2 * radius + 1
        # create the mask for the application of the values
        dimx = int(self.width)
        dimy = int(self.height)
        mask2 = np.full((dimx, dimy), False, dtype=bool)
        maskp = self.mask[max(0, -x):min(maskdim, dimx-x), max(0, -y):min(maskdim,dimy-y)]
        # print(f"maskp = {maskp}")
        mask2[max(0, x):min(dimx, x+maskdim), max(0, y):min(dimy,y+maskdim)] = maskp
        value[mask2] = new_value
        # logging.info("apply_value_mask done")


## Functions used for the visualization

def visualizeEnvAndIM(ax_env, ax_im_value, ax_im_uncertainty, env, im):
    score = im.score(env)
    ax_env.set_title("Environment")
    image_env = ax_env.imshow(env.value, vmin=0, vmax=5.0)
    ax_im_value.set_title(f"IM.value sc={score:0.2}")
    image_im_value = ax_im_value.imshow(im.value, vmin=0, vmax=5.0)
    ax_im_uncertainty.set_title("IM - uncertainty")
    image_im_uncertainty = ax_im_uncertainty.imshow(im.uncertainty, vmin=0)
    # vmax=1.0 


if __name__ == "__main__":
    if True:
        # create an environment to observe
        env = DissipationModelEnvironment("water", 100, 100, seed=1)
        env.evolve_speed = 1
        env.p_pollution = 0.1
        for t in range(120):
            env.proceed()
        #plt.imshow(env.value, vmin=0, vmax=1.0)

        im = DiskEstimateScalarFieldIM("sample", env.width, env.height)
        # im = GaussianProcessScalarFieldIM("sample", env.width, env.height)


        # generate a series random observations
        scores = []
        times = []
        for i in range(150):
            x = random.randint(0, env.width-1)
            y = random.randint(0, env.height-1)
            value = env.value[x,y]
            obs = {"x": x, "y": y, "value": value}
            im.add_observation(obs)
            time = timeit.timeit("im.proceed(1)", number=1,  globals=globals())
            times.append(time)
            scores.append(im.score(env))

        fig, ((ax_env, ax_im_value, ax_im_uncertainty, ax_score, ax_time)) = plt.subplots(1,5, figsize=(15,3))
        visualizeEnvAndIM(ax_env, ax_im_value, ax_im_uncertainty, env, im)


        ax_score.set_title("Score (avg)")
        ax_score.plot(scores)
        ax_time.set_title("Estimation time (sec)")
        ax_time.plot(times)
        plt.tight_layout()
        plt.show()