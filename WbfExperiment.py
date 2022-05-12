from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy, InformationGreedyPolicy
from PathGenerators import generate_lawnmower
from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, waterberry_score
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

import bz2 as compress

# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# ***************************************************************
#
#  The standard experiments of the WBF benchmark. 
#  Call the menu function (interactively or with parameters), and 
#  then visualize and analyze the returned result. 
#
# ***************************************************************

def visualize_1S(results, savedir):
    """Visualizes the results of a non-oneshot run for a 1S run. All the information is taken from the results dictionary"""
    fig, ((ax_env_tylcv, ax_im_tylcv, ax_obs, ax_scores)) = plt.subplots(1, 4, figsize=(20,5))
    _visualize_1S(ax_env_tylcv, ax_im_tylcv, ax_obs, results)
    ax_scores.plot(results["scores"])
    ax_scores.set_xlabel("Time")
    ax_scores.set_ylabel("Score")
    plt.savefig(pathlib.Path(savedir, "iterative.pdf"))


def visualize_1S_oneshot(results, savedir):
    """Visualizes the results of a oneshot run for a 1S run. All the information is taken from the results dictionary"""
    fig, ((ax_env_tylcv, ax_im_tylcv, ax_obs)) = plt.subplots(3, figsize=(5,15))
    _visualize_1S(ax_env_tylcv, ax_im_tylcv, ax_obs, results)
    plt.savefig(pathlib.Path(savedir, "oneshot.pdf"))

def _visualize_1S(ax_env_tylcv, ax_im_tylcv, ax_obs, results):
    """The shared part of the 1S runs"""
    wbfe = results["wbfe"]
    wbfim = results["wbfim"]
    image_env_tylcv = ax_env_tylcv.imshow(wbfe.tylcv.value.T, vmin=0, vmax=1, origin="lower", cmap="gray")
    ax_env_tylcv.set_title("Environment at end of scenario")
    image_im_tylcv = ax_im_tylcv.imshow(wbfim.im_tylcv.value.T, vmin=0, vmax=1, origin="lower", cmap="gray")
    ax_im_tylcv.set_title("Information model at end of scenario")
    obsx = []
    obsy = []
    for obs in results["observations"]:
        obsx.append(obs[StoredObservationIM.X])
        obsy.append(obs[StoredObservationIM.Y])
    ax_obs.set_title("Observations")
    ax_obs.plot(obsx, obsy)
    ax_obs.set_xlim(xmin = 0, xmax = wbfe.width)
    ax_obs.set_ylim(ymin = 0, ymax = wbfe.height)


def simulate_1day(results):
    """Runs the simulation for one day. All the parameters and output is in the results dictionary"""
    wbfe, wbf = results["wbfe"], results["wbf"]
    results["wbfim"] = wbfim = WaterberryFarmInformationModel("wbfi", wbf.width, wbf.height)
    positions = []
    observations = []
    scores = []
    for time in range(int(results["timespan"])):
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
            score = waterberry_score(wbfe, wbfim)
            scores.append(score)
    if results["oneshot"]:
        wbfim.proceed(1)
        score = waterberry_score(wbfe, wbfim)
    results["score"] = score
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions


def simulate_15day(results):
    """Runs the simulation for 15 days. All the parameters and output is in the results dictionary"""
    wbfe, wbf = results["wbfe"], results["wbf"]
    results["wbfim"] = wbfim = WaterberryFarmInformationModel("wbfi", wbf.width, wbf.height)
    positions = []
    observations = []
    scores = []
    for days in range(15):
        for time in range(int(results["timespan"])):
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
                score = waterberry_score(wbfe, wbfim)
                scores.append(score)
        wbfe.proceed(1)
    if results["oneshot"]:
        wbfim.proceed(1)
        score = waterberry_score(wbfe, wbfim)
    results["score"] = score
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions

def menu(choices):
    """Running the simulator in an interactive mode. The choices dictionary 
    can contain some predefined choices, which allow the bypassing of the menus, including the fully automated run. 
    Returns the results in the form of dictionary, which is also saved."""
    results = {}
    results["choices"] = choices

    if "geometry" not in choices:
        print("Choose the geometry:")
        print("\t1. Miniberry-10:")
        print("\t2. Miniberry-30:")
        print("\t3. Miniberry-100:")
        print("\t4. Waterberry")
        choices["geometry"] = int(input("Choose desired geometry: "))
    if choices["geometry"] == 1:
        results["typename"] = "Miniberry-10"
    elif choices["geometry"] == 2:
        results["typename"] = "Miniberry-30"
    elif choices["geometry"] == 3:
        results["typename"] = "Miniberry-100"
    elif choices["geometry"] == 4:
        results["typename"] = "Waterberry"
    else:
        raise Exception(f"Unknown choice of geometry {choices['geometry']}")

    if "action" not in choices:
        print("Choose the action:")
        print("\t1. Precompute the environment values")
        print("\t2. Visualize the environment at a given time")
        print("\t3. Show an animation of the environment evolution")
        print("\t4. Run an experiment with a given robot policy")
        print("\t5. Generate graphs from previously run experiments")
        choices["action"] = int(input("Choose desired action: "))
    if choices["action"] == 1:
        wbf, wbfe, savedir = create_wbfe(False, wbf_prec = None, typename = results["typename"])
        for i in range(200):
            logging.info(f"precalculation proceed {i}")
            wbfe.proceed()
        return results        
    elif choices["action"] == 2:
        wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
        results["wbf"] = wbf
        results["wbfe"] = wbfe
        results["savedir"] = savedir
        timepoint = int(input("Timepoint for visualization:"))
        wbfe.proceed(timepoint)
        wbfe.visualize()
        plt.show()
        return results
    elif choices["action"] == 3:
        wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
        wbfe.visualize()
        anim = wbfe.animate_environment()
        plt.show()
        return results
    elif choices["action"] == 4:
        if "time_start_environment" not in choices:
            choices["time_start_environment"] = int(input("Time in the environment when the robot starts: "))
        results["time_start_environment"] = choices["time_start_environment"] # FIXME: dependent...
        wbf, wbfe, savedir = create_wbfe(saved=True, wbf_prec=None, typename = results["typename"])
        wbfe.proceed(results["time_start_environment"])
        wbfe.visualize()
        results["wbf"] = wbf
        results["wbfe"] = wbfe
        results["savedir"] = savedir
    elif choices["action"] == 5:
        print("NOT IMPLEMENTED YET: generate graphs from previously run experiments")
    else:
        raise Exception(f"Unknown choice of action {choices['action']}")

    # If we are here, we are running an experiment
    exp_name = results["typename"] + "_" # name of the exp., also dir to save data
    if "scenario" not in choices:
        print("Choose the experiment scenario")
        print("\t1. Single day, static, TYLCV only")
        print("\t2. Single day, static, multi-value")
        print("\t3. 15 day, dynamic, TYLCV only")
        print("\t4. 15 day, dynamic, multi-value")
        choices["scenario"] = int(input("Choose desired experiment scenario: "))
    
    results["scenario"] = choices["scenario"]
    if choices["scenario"] == 1:
        results["days"] = 1
        results["values"] = "single"
        exp_name = exp_name + "1S_"
    elif choices["scenario"] == 2:
        results["days"] = 1
        results["values"] = "multi"
        exp_name = exp_name + "1M_"
    elif choices["scenario"] == 3:
        results["days"] = 15
        results["values"] = "single"
        exp_name = exp_name + "15S_"
    elif choices["scenario"] == 4:
        results["days"] = 15
        results["values"] = "multi"
        exp_name = exp_name + "15M_"
    else:
        raise Exception(f"Unknown choice of scenario {choices['scenario']}")

    if "policy" not in choices:
        print("Choose the policy")
        print("\t1. Lawnmower - full coverage by day")
        print("\t2. Lawnmower - restart")
        print("\t3. Random waypoint", flush = True)
        choices["policy"] = int(input("Choose desired policy: "))
    # setting up the run

    results["robot"] = Robot("Rob", 0, 0, 0, env=None, im=None)
    if results["typename"] == "Miniberry-10":
        xmin, xmax, ymin, ymax = 0, 10, 0, 10
        timespan = 0.4 * 100 
    elif results["typename"] == "Miniberry-30":
        xmin, xmax, ymin, ymax = 0, 30, 0, 30
        timespan = 0.4 * 900
    elif results["typename"] == "Miniberry-100":
        xmin, xmax, ymin, ymax = 0, 100, 0, 100
        timespan = 0.4 * 10000
    elif results["typename"] == "Waterberry":
        xmin, xmax, ymin, ymax = 1000, 5000, 1000, 4000
        timespan = 0.4 * 12000000
    results["velocity"] = 1
    results["timespan"] = timespan

    if choices["policy"] == 1: # lawnmower that covers the area in one day
        # FIXME: carefully check the parameters here
        results["policy"] = "lawnmower-lowres"
        exp_name = exp_name + results["policy"]
        path = generate_lawnmower(xmin, xmax, ymin, ymax, winds = 5)
        results["robot"].policy = FollowPathPolicy(None, results["robot"], vel = results["velocity"], waypoints = path, repeat = True)        
    elif choices["policy"] == 2: # lawnmower that restarts... 
        # FIXME: carefully check the parameters here
        results["policy"] = "lawnmower-restart"
        exp_name = exp_name + results["policy"]
        path = generate_lawnmower(xmin, xmax, ymin, ymax, winds = 50)
        results["robot"].policy = FollowPathPolicy(None, results["robot"], vel = results["velocity"], waypoints = path, repeat = True)        
    elif choices["policy"] == 3: # random waypoint
        results["policy"] = "random-waypoint"
        exp_name = exp_name + results["policy"]
        results["robot"].policy = RandomWaypointPolicy(results["wbfe"], results["robot"], vel = results["velocity"], low_point = [xmin, ymin], high_point = [xmax, ymax], seed = 0)        
    else: 
        raise Exception(f"Unknown policy choice {choices['policy']}")

    # if we are here, we are running the experiment

    if results["days"] == 1:
        results["oneshot"] = False # calculate one observation score for all obs.
        simulate_1day(results)
    elif results["values"] == "multi" and results["days"] == 1:
        raise Exception("NOT IMPLEMENTED YET: multi-value single-day scenario")
    elif results["values"] == "single" and results["days"] == 15:
        results["oneshot"] = False # calculate one observation score for all obs.
        simulate_15day(results)
    elif results["values"] == "multi" and results["days"] == 15:
        raise Exception("NOT IMPLEMENTED YET: multi-value multi-day scenario")
    else:
        raise Exception(f"Unsupported combination of values {results['values']}and days results['days'] in the scenario")




    # FIXME: The savedir is not the right one for this!!!! especially if
    # this is being run from BenchmarkPaper-experiments.ipnb
    # Encode the choices
    resultsFileName = f"res-pol_{choices['policy']}"
    resultsPath = pathlib.Path(results["savedir"], resultsFileName)
    logging.info(f"Saving results to: {resultsPath}")
    with compress.open(resultsPath, "wb") as f:
        pickle.dump(results, f)

    if "visualize" not in choices:
        choices["visualize"] = int(input("Visualize? (1 yes, 0 no): "))
    if choices["visualize"] == 1:
        print("Here!!!", flush=True)
        if results["days"] == 1:
            if results["oneshot"]:
                visualize_1S_oneshot(results, savedir)
            else:
                visualize_1S(results, savedir)
            plt.show()
        elif results["values"] == "multi" and results["days"] == 1:
            raise Exception("NOT IMPLEMENTED YET: multi-value single-day scenario")
        elif results["values"] == "single" and results["days"] == 15:
            visualize_1S(results, savedir)
            if results["oneshot"]:
                visualize_1S_oneshot(results, savedir)
            else:
                visualize_1S(results, savedir)
            plt.show()
        elif results["values"] == "multi" and results["days"] == 15:
            raise Exception("NOT IMPLEMENTED YET: multi-value multi-day scenario")
        else:
            raise Exception(f"Unsupported combination of values {results['values']}and days results['days'] in the scenario")
    return results

if __name__ == "__main__":
    results = menu({})
