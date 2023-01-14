from secrets import choice
from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy, InformationGreedyPolicy
from PathGenerators import find_fixed_budget_spiral, generate_lawnmower, find_fixed_budget_lawnmower, generate_spiral_path, find_fixed_budget_spiral
from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import animation
import numpy as np
import pathlib
import logging
import pickle
import copy

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



def simulate_1day(results):
    """Runs the simulation for one day. All the parameters and output is in the results dictionary. It uses the WaterberryFarmInformationModel for modeling and the waterberry_score() for scoring"""
    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    positions = []
    observations = []
    scores = []
    for time in range(int(results["timesteps-per-day"])):
        # print(f"Simulate_1day I am at time = {time}")
        results["robot"].enact_policy()
        results["robot"].proceed(1)
        position = [int(results["robot"].x), int(results["robot"].y), time]
        # print(results["robot"])
        positions.append(position)
        obs = wbfe.get_observation(position)
        observations.append(obs)
        wbfim.add_observation(obs)
        if not results["oneshot"]:
            wbfim.proceed(1)
            score = results["score-code"].score(wbfe, wbfim)
            scores.append(score)
    if results["oneshot"]:
        wbfim.proceed(1)
        score = results["score-code"].score(wbfe, wbfim)
    results["score"] = score
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions


def simulate_15day(results):
    """Runs the simulation for 15 days. All the parameters and output is in the results dictionary"""
    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    positions = []
    observations = []
    scores = []
    for days in range(15):
        for time in range(int(results["timesteps-per-day"])):
            results["robot"].enact_policy()
            results["robot"].proceed(1)
            position = [int(results["robot"].x), int(results["robot"].y), time]
            print(results["robot"])
            positions.append(position)
            obs = wbfe.get_observation(position)
            observations.append(obs)
            wbfim.add_observation(obs)
            if not results["oneshot"]:
                wbfim.proceed(1)
                score = results["score-code"].score(wbfe, wbfim)
                scores.append(score)
        wbfe.proceed(1)
    if results["oneshot"]:
        wbfim.proceed(1)
        score = results["score-code"].score(wbfe, wbfim)
    results["score"] = score
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions


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

def action_run_oneday(choices):
    results = copy.deepcopy(choices)
    results["action"] = "run-one-day"
    menuGeometry(results)
    if "time-start-environment" not in results:
        results["time-start-environment"] = int(input("Time in the environment when the robot starts: "))
    wbf, wbfe, savedir = create_wbfe(saved=True, wbf_prec=None, typename = results["typename"])
    wbfe.proceed(results["time-start-environment"])
    # wbfe.visualize()
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
    if "result-basedir" in results:
        results_path = pathlib.Path(results["result-basedir"], results_filename)
    else:
        results_path = pathlib.Path(results["savedir"], results_filename)    
    results["results-path"] = results_path
    # if dryrun is specified, we return the results without running anything
    if "dryrun" in results and results["dryrun"] == True:
        return results
    results["oneshot"] = False # calculate one observation score for all obs.
    # running the simulation
    simulate_1day(results)
    #
    logging.info(f"Saving results to: {results_path}")
    with compress.open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results



if __name__ == "__main__":
    results = menu({})
