# Waterberry Farms Experiments: functions helping to run experiments
# with the Waterberry Farms benchmark

from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy
from PathGenerators import find_fixed_budget_spiral, generate_lawnmower, find_fixed_budget_lawnmower, generate_spiral_path, find_fixed_budget_spiral
from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, WBF_MultiScore

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

def get_geometry(typename, geo = None):
    """Returns an object with the geometry for the different types (or adds it into the passed dictionary"""
    if geo == None:
        geo = {}
    geo["velocity"] = 1

    if typename == "Miniberry-10":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 0, 10, 0, 10
        geo["width"], geo["height"] = 11, 11
        geo["timesteps-per-day"] = 0.4 * 100 
    elif typename == "Miniberry-30":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 0, 30, 0, 30
        geo["width"], geo["height"] = 31, 31
        geo["timesteps-per-day"] = 0.4 * 900
    elif typename == "Miniberry-100":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 0, 100, 0, 100
        geo["width"], geo["height"] = 101, 101
        geo["timesteps-per-day"] = 0.4 * 10000
    elif typename == "Waterberry":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 1000, 5000, 1000, 4000
        geo["width"], geo["height"] = 5001, 4001
        geo["timesteps-per-day"] = 0.4 * 12000000
    return geo

def menuGeometry(results):
    """Asks about the geometry and sets it in results"""
    if "geometry" not in results:
        print("Choose the geometry:")
        print("\t1. Miniberry-10:")
        print("\t2. Miniberry-30:")
        print("\t3. Miniberry-100:")
        print("\t4. Waterberry")
        results["geometry"] = int(input("Choose desired geometry: "))
    if results["geometry"] == 1 or results["geometry"] == "Miniberry-10":
        results["typename"] = "Miniberry-10"
    elif results["geometry"] == 2 or results["geometry"] == "Miniberry-30":
        results["typename"] = "Miniberry-30"
    elif results["geometry"] == 3 or results["geometry"] == "Miniberry-100":
        results["typename"] = "Miniberry-100"
    elif results["geometry"] == 4 or results["geometry"] == "Waterberry":
        results["typename"] = "Waterberry"
    else:
        raise Exception(f"Unknown choice of geometry {results['geometry']}")


def menu_scenario(choices):
    """Interactively selecting the scenario and adding it into choices"""
    print("Choose the experiment scenario")
    print("\t1. Single day, static, TYLCV only")
    print("\t2. Single day, static, multi-value")
    print("\t3. 15 day, dynamic, TYLCV only")
    print("\t4. 15 day, dynamic, multi-value")
    choice = int(input("Choose desired experiment scenario: "))
    if choice == 1:
        choices["scenario"] = "one-day-single-value"
    elif choice == 2:
        choices["scenario"] = "one-day-multi-value"
    elif choice == 3:
        choices["scenario"] == "15-day-single-value"
    elif choice == 4:
        choices["scenario"] == "15-day-multi-value"

def action_precompute_environment(choices):
    results = copy.deepcopy(choices)
    results["action"] = "precompute-environment"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(False, wbf_prec = None, typename = results["typename"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    if "precompute-time" in results:
        for i in range(results["precompute-time"]):
            logging.info(f"precalculation proceed {i}")
            wbfe.proceed()
    return results 

def action_load_environment(choices):
    results = copy.deepcopy(choices)
    results["action"] = "load-environment"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    if "precompute-time" in results:
        for i in range(results["precompute-time"]):
            logging.info(f"precalculation proceed {i}")
            wbfe.proceed()
    return results 

def action_visualize(choices):
    results = copy.deepcopy(choices)
    results["action"] = "visualize"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    timepoint = int(input("Timepoint for visualization:"))
    wbfe.proceed(timepoint)
    wbfe.visualize()
    plt.show()
    return results

def action_animate(choices):
    results = copy.deepcopy(choices)
    results["action"] = "visualize"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
    wbfe.visualize()
    anim = wbfe.animate_environment()
    plt.show()
    return results

def simulate_1day(results):
    """Runs the simulation for one day. All the parameters and output is in the results dictionary."""
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


def action_run_oneday(choices):
    results = copy.deepcopy(choices)
    results["action"] = "run-one-day"
    if "im_resolution" not in results:
        results["im_resolution"] = 1
    menuGeometry(results)
    if "time-start-environment" not in results:
        results["time-start-environment"] = int(input("Time in the environment when the robot starts: "))
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
    results["robot"] = Robot("Rob", 0, 0, 0, env=None, im=None)
    # Setting the policy
    results["exp-name"] = results["exp-name"] + results["policy-name"]
    results["robot"].assign_policy(results["policy-code"])

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
    simulate_1day(results)
    #
    logging.info(f"Saving results to: {results_path}")
    with compress.open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results

def simulate_15day(results):
    """Runs the simulation for 15 days. All the parameters and output is in the results dictionary"""
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
    for day in range(1,16): # 1,2,...15 day zero is the beginning
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


def action_run_multiday(choices):
    results = copy.deepcopy(choices)
    results["action"] = "run-15-day"
    menuGeometry(results)
    if "time-start-environment" not in results:
        results["time-start-environment"] = int(input("Time in the environment when the robot starts: "))
    wbf, wbfe, savedir = create_wbfe(saved=True, wbf_prec=None, typename = results["typename"])
    wbfe.proceed(results["time-start-environment"])
    results["wbf"] = wbf

    # wbfe will be the environment at the end of experiment
    results["wbfe"] = wbfe
    results["wbfe-days"] = {}
    results["wbfe-days"][0] = copy.deepcopy(wbfe)

    results["savedir"] = savedir
    results["exp-name"] = results["typename"] + "_" # name of the exp., also dir to save data
    results["days"] = 15
    results["exp-name"] = results["exp-name"] + "1M_"
    get_geometry(results["typename"], results)
    # override the velocity and the timesteps per day, if specified
    if "timesteps-per-day-override" in results:
        results["timesteps-per-day"] = results["timesteps-per-day-override"]
    if "velocity-override" in results:
        results["velocity"] = results["velocity-override"]
    results["robot"] = Robot("Rob", 0, 0, 0, env=None, im=None)
    # Setting the policy
    results["exp-name"] = results["exp-name"] + results["policy-name"]
    results["robot"].assign_policy(results["policy-code"])

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
    results["oneshot"] = False # calculate one observation score for all obs.
    # running the simulation
    simulate_15day(results)
    #
    logging.info(f"Saving results to: {results_path}")
    with compress.open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results

