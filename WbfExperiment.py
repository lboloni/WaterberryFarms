"""
WbfExperiment.py

Waterberry Farms Experiments: functions helping to run experiments
with the Waterberry Farms benchmark.


As of 2025-03-21 this will be (gradually) refactored to use the Experiment/Run 
configuration framework.

"""



# from environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
# from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
# from Robot import Robot
# from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
#     AbstractWaypointPolicy
# from PathGenerators import find_fixed_budget_spiral, generate_lawnmower, find_fixed_budget_lawnmower, generate_spiral_path, find_fixed_budget_spiral
# from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, WBF_MultiScore
# from wbf_helper import get_geometry

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


def simulate_1day(results):
    """
    Runs the simulation for one day. All the parameters and output is in the results dictionary.
    As this can also be slow, it performs time tracking, and marks the time tracking values into the results as well.
    """
    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    
    # we are assigning here to the robot the information model
    # FIXME: this might be more tricky later
    results["robot"].im = wbfim
    
    positions = []
    observations = []
    scores = []
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions
    results["computation-cost-policy"] = []

    time_track_start = time.time_ns()
    time_track_last_start = time_track_start

    im_resolution_count = 0

    for timestep in range(int(results["timesteps-per-day"])):
        # print(f"Simulate_1day I am at time = {time}")

        time_policy_start = time.time_ns()

        results["robot"].enact_policy()
        results["robot"].proceed(1)
        position = [int(results["robot"].x), int(results["robot"].y), timestep]
        # print(results["robot"])
        positions.append(position)
        obs = wbfe.get_observation(position)
        observations.append(obs)
        wbfim.add_observation(obs)
        results["robot"].add_observation(obs)
        time_policy_finish = time.time_ns()
        results["computation-cost-policy"].append(time_policy_finish - time_policy_start)

        # update the im and the score at every im_resolution steps and at the last iteration
        im_resolution_count += 1
        if im_resolution_count == results["im_resolution"] or timestep + 1 == results["timesteps-per-day"]:
            wbfim.proceed(results["im_resolution"])
            score = results["score-code"].score(wbfe, wbfim)
            for i in range(im_resolution_count):
                scores.append(score)
            im_resolution_count = 0
        if "hook-after-day" in results:
            results["hook-after-day"](results)
        # every 10 seconds, write out where we are
        time_track_current = time.time_ns()
        if (time_track_current - time_track_last_start) > 10e9:
            print(f"At {timestep} / {int(results['timesteps-per-day'])} elapsed {int((time_track_current - time_track_start) / 1e9)} seconds")
            time_track_last_start = time_track_current
    results["score"] = score



def simulate_multiday(results):
    """
    Runs the simulation for multiple days. All the parameters and output 
    are in the results dictionary
    """
    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    results["wbfim-days"] = {}
    results["wbfim-days"][0] = copy.deepcopy(wbfim)
    # keep the positions and remember the length to split by days
    positions = []
    positions_size_last = len(positions)
    results["positions-days"] = {}
    # keep the observations and remember the lenght to split by days    
    observations = []
    observations_size_last = len(observations)
    results["observations-days"] = {}    
    # keep the scores and remember the lenght to split by days    
    scores = []
    scores_size_last = len(scores)
    results["scores-days"] = {}
    for day in range(1,results["days"]): # 1,2,...15 day zero is the beginning
        print(f"day {day}")
        elapsedCount = 0 # how many steps elapsed since we updated im
        for time in range(int(results["timesteps-per-day"])):
            results["robot"].enact_policy()
            results["robot"].proceed(1)
            position = [int(results["robot"].x), int(results["robot"].y), time]
            positions.append(position)
            obs = wbfe.get_observation(position)
            observations.append(obs)
            wbfim.add_observation(obs)
            # determine if we perform the calculation and for how many days
            # FIXME: debug this
            performCount = 0            
            if time == 0: # first time
                performCount = 1
            elif elapsedCount % int(results["score-resolution"]) == 0:
                performCount = elapsedCount
                elapsedCount = 0
            elif time == int(results["timesteps-per-day"])-1:
                performCount = elapsedCount
                elapsedCount = 0
            ## if performcount is not zero update it.
            if performCount > 0:
                wbfim.proceed(performCount)
                score = results["score-code"].score(wbfe, wbfim)
                for i in range(performCount):
                    scores.append(score)
            elapsedCount += 1

        # store the values at the end of the day
        results["wbfe-days"][day] = copy.deepcopy(wbfe)
        results["wbfim-days"][day] = copy.deepcopy(wbfim)
        results["positions-days"][day] = positions[positions_size_last:]
        positions_size_last = len(positions)
        results["observations-days"][day] = observations[observations_size_last:]
        observations_size_last = len(observations)
        results["scores-days"][day] = scores[scores_size_last:]
        scores_size_last = len(scores)
        # move to the next day in the environment
        wbfe.proceed(1)
    # store in results the values at the end of scenario
    results["score"] = score
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions


