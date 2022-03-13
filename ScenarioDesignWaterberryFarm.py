import Environment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from WaterberryFarm import WaterberryFarm, MiniberryFarm, WaterberryFarmEnvironment
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
import pathlib 
import pickle
import copy
import random

import logging
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# This file is used to interactively design the scenario for the 
# waterberry farm. We are calibrating the parameters and initial infections in 
# such a way as to have an interesting scenario for strawberry, tomato and soil humidity.

# Tomato scenario
# We are going to assume that one step is one day

def create_wbf():
    """Creates an empty farm and environment"""
    p = pathlib.Path.cwd()
    savedir = pathlib.Path(p.parent, "__Temporary", p.name + "_data", "saved")
    path_geometry = pathlib.Path(savedir,"farm_geometry")
    path_environment = pathlib.Path(savedir,"farm_environment")
    if path_geometry.exists():
        logging.info("loading the geometry and environment from saved data")
        with open(path_geometry, "rb") as f:
            wbf = pickle.load(f)
        with open(path_environment, "rb") as f:
            wbfe = pickle.load(f)
        logging.info("loading done")
        return wbf, wbfe
    wbf = WaterberryFarm()
    wbf = MiniberryFarm(scale=10)
    wbf.create_type_map()
    wbfe = WaterberryFarmEnvironment(wbf, seed = 10)
    # ************ save the geometry and environment
    savedir.mkdir(parents=True, exist_ok=True)
    with open(path_geometry, "wb+") as f:
        pickle.dump(wbf, f)
    with open(path_environment, "wb+") as f:
        pickle.dump(wbfe, f)
    return wbf, wbfe

def apply_transformation(wbfe, parameters):
    """Applies the transformation described in the parameters"""
    logging.info(f"apply transformation started {parameters}")
    wbfe.tylcv.change_p_transmission(parameters["p_transmission"])
    # need to recalculate the infection matrix
    wbfe.tylcv.p_infection = wbfe.tylcv.calculate_infection_matrix(wbfe.tylcv.p_transmission, wbfe.tylcv.spread_dimension)
    wbfe.tylcv.infection_duration = int(parameters["infection_duration"])
    # place some infections
    for i in range(int(parameters["infection_count"])):
        locationx = int( wbfe.tylcv.random.random(1) * wbfe.tylcv.width )
        locationy = int ( wbfe.tylcv.random.random(1) * wbfe.tylcv.height )
        wbfe.tylcv.status[locationx, locationy] = int(parameters["infection_value"])
    for i in range(int(parameters["proceed_count"])):
        wbfe.tylcv.proceed(1)
        print(f"{i}")
    logging.info("apply transformation done")
    return wbfe

def mutate_parameters(parameters, names, ratio = 0.5, count = 4):
    """Starting from a certain parameter, creates a list of new parameters"""
    list = [parameters]
    for name in names:
        newlist = []
        for par in list:
            newlist.extend(mutate_parameter(par, name, ratio))
        list = newlist
    return random.sample(list, count)       

def mutate_parameter(parameters, name, scale):
    par = copy.deepcopy(parameters)
    par_plus = copy.deepcopy(parameters)
    par_plus[name] = par_plus[name] * (1.0 + scale)
    par_minus = copy.deepcopy(parameters)
    par_minus[name] = par_minus[name] * (1.0 - scale)
    return [par_minus, par, par_plus]

def search_for_tylcv():
    parameters = {"infection_count": 3, "infection_value": 5, "proceed_count": 25, "p_transmission": 0.5, "infection_duration": 7}
    while True:
        #names = list(parameters.keys())
        names = ["infection_count", "p_transmission"]
        pars = mutate_parameters(parameters, names)
        wbfes = []
        print(pars)
        for par in pars:
            wbf, wbfe = create_wbf()
            wbfet = apply_transformation(wbfe, par)
            wbfes.append(wbfet)
        # now visualize 
        fig, axes = plt.subplots(len(wbfes), figsize=(5,20))
        for i, wbfet in enumerate(wbfes):
            axes[i].imshow(wbfet.tylcv.value.T, vmin = 0, vmax = 1, origin = "lower")
            axes[i].set_title(f"Choice {i}")
        plt.show()
        choice = int(input("Choose the best: "))
        parameters = pars[choice]
        print(f"Current best parameters: {parameters}")


if __name__ == "__main__":
    # search_for_tylcv()
    p = pathlib.Path.cwd()
    savedir = pathlib.Path(p.parent, "__Temporary", p.name + "_data")
    print(p.name)
    print(p.parent)
    print(savedir)