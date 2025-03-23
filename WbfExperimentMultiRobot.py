"""
WbfExperimentMultiRobot.py

WBF: experimental harness for multi-robot experiments. 
Very similar to the WbfExperiments, where are the single robot 
experiments. 

As of 2025-03-21 this will be refactored to use the Experiment/Run 
configuration framework.

"""
from environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy
from PathGenerators import find_fixed_budget_spiral, generate_lawnmower, find_fixed_budget_lawnmower, generate_spiral_path, find_fixed_budget_spiral
from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, WBF_MultiScore
#from WbfExperiment import get_geometry, menuGeometry
from wbf_helper import get_geometry

import numpy as np
import pathlib
import logging
import pickle
import copy
import time

#import bz2 as compress
import gzip as compress

# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

def simulate_1day_multirobot(results):
    """
    Runs the simulation for one day. All the parameters and output is in the results dictionary.
    As this can also be slow, it performs time tracking, and marks the time tracking values into the results as well.
    """
    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    
    # we are assigning here to all the robots the information model
    # FIXME: this might be more tricky later
    for robot in results["robots"]:
        robot.im = wbfim
    
    # positions, observations and scores are all list of lists
    # first index: time, second index: robot
    results["scores"] = []
    results["observations"] = []
    results["positions"] = []
    results["computation-cost-policy"] = []

    time_track = {}
    time_track["start"] = time.time_ns()
    time_track["last_start"] = time_track["start"]

    im_resolution_count = 0 # counting the steps for the im update

    for timestep in range(int(results["timesteps-per-day"])):
        time_track["policy_start"] = time.time_ns()

        positions = []
        observations = []

        for robot in results["robots"]:
            robot.enact_policy()
            robot.proceed(1)
            position = [int(robot.x), int(robot.y), timestep]
            obs = wbfe.get_observation(position)
            wbfim.add_observation(obs)
            robot.add_observation(obs)
            positions.append(position)
            observations.append(obs)

        results["positions"].append(positions)
        results["observations"].append(observations)

        time_track["policy_finish"] = time.time_ns()
        results["computation-cost-policy"].append(time_track["policy_finish"] - time_track["policy_start"])

        # update the im and the score at every im_resolution steps and at the last iteration
        im_resolution_count += 1
        if im_resolution_count == results["im_resolution"] or timestep + 1 == results["timesteps-per-day"]:
            wbfim.proceed(results["im_resolution"])
            score = results["score-code"].score(wbfe, wbfim)
            for i in range(im_resolution_count):
                results["scores"].append(score)
            im_resolution_count = 0
        if "hook-after-day" in results:
            results["hook-after-day"](results)
        # every 10 seconds, write out where we are
        time_track["current"] = time.time_ns()
        if (time_track["current"] - time_track["last_start"]) > 10e9:
            print(f"At {timestep} / {int(results['timesteps-per-day'])} elapsed {int((time_track['current'] - time_track['start']) / 1e9)} seconds")
            time_track["last_start"] = time_track["current"]
    # the final score
    results["score"] = score


