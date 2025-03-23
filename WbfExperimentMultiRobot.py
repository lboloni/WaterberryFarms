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


def action_run_1day_multirobot(choices):
    """
    Implements a single-day experiment with a multiple robots in the WBF simulator. This is the top level function which is called to run the experiment. 

    choices: a dictionary into which the parameters of the experiments are being loaded. A copy of this will be internally created. 

    results: the return value, which contains both all the input values in the choices, as well as the output data (or references to it.)

    """
    results = copy.deepcopy(choices)
    results["action"] = "run-one-day-multirobot"
    if "im_resolution" not in results:
        results["im_resolution"] = 1
    menuGeometry(results)
    if "time-start-environment" not in results:
        results["time-start-environment"] = int(input("Time in the environment when the robots start: "))
    wbf, wbfe, savedir = create_wbfe(saved=True, wbf_prec=None, typename = results["typename"])
    wbfe.proceed(results["time-start-environment"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    results["exp-name"] = results["typename"] + "_" # name of the exp., also dir to save data
    results["days"] = 1
    results["exp-name"] = results["exp-name"] + "1M_"
    get_geometry(results["typename"], results)
    # override the velocity and the timesteps per day, if specified
    if "timesteps-per-day-override" in results:
        results["timesteps-per-day"] = results["timesteps-per-day-override"]
    if "velocity-override" in results:
        results["velocity"] = results["velocity-override"]

    # creating the number of robots. The number of robots is specified by the number of policies that we pass

    robots = []
    for p in results["policy-code"]:
        robot = Robot("Rob", 0, 0, 0, env=None, im=None)
        robot.assign_policy(p)
        robots.append(robot)
        results["exp-name"] = results["exp-name"] + "_" + p.name

    results["robots"] = robots
 
    if "results-filename" in results:
        results_filename = results["results-filename"]
    else:
        results_filename = f"res_{results['exp-name']}"
    if "results-basedir" in results:
        results_path = pathlib.Path(results["results-basedir"], results_filename)
    else:
        results_path = pathlib.Path(results["savedir"], results_filename)    
    results["results-path"] = results_path
    # if dryrun is specified, we return the results without running anything
    if "dryrun" in results and results["dryrun"] == True:
        return results
    # results["oneshot"] = False # calculate one observation score for all obs.
    # running the simulation
    simulate_1day_multirobot(results)
    #
    logging.info(f"Saving results to: {results_path}")
    with compress.open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results
