from secrets import choice
from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy
from PathGenerators import find_fixed_budget_spiral, generate_lawnmower, find_fixed_budget_lawnmower, generate_spiral_path, find_fixed_budget_spiral
from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, WBF_MultiScore
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import animation
import matplotlib.lines as lines

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

def graph_robot_path_day(results, day, ax):
    """visualize the observations, which gives us the path of the robot"""
    # FIXME: what about the positions?
    wbfe = results["wbfe"]
    empty = np.ones_like(wbfe.tylcv.value.T)
    image_env_tylcv = ax.imshow(empty, vmin=0, vmax=1, origin="lower", cmap="gray")    
    ax.set_title("Robot path")
    obsx = []
    obsy = []
    for obs in results["observations-days"][day]:
        obsx.append(obs[StoredObservationIM.X])
        obsy.append(obs[StoredObservationIM.Y])
        old_obs = obs
    ax.add_line(lines.Line2D(obsx, obsy, color="red"))

def graph_scores_per_day(results, ax):
    """Plot the scores on a one-a-day basis in the results of a multi-day experiment"""
    # visualize the observations, which gives us the path of the robot
    score_to_plot = []
    days_to_plot = []
    for day in range(1, 16):
        scores = results["scores-days"][day]
        score_to_plot.append(scores[-1])
        days_to_plot.append(day)
    ax.plot(days_to_plot, score_to_plot)
    ax.set_ylim(top=0)
    ax.set_xlabel("Days")
    ax.set_title("Scores")

def graph_scores(results, ax, label = None):
    """Plot the scores, for the scores that return a single value"""
    if label is None:
        scores = results["scores"]
        ax.set_title("Scores")
    else:   
        scores = [a[label] for a in results["scores"]]
        ax.set_title(f"Score {label}")
    ax.plot(scores)
    ax.set_ylim(top=0)
    ax.set_xlabel("Time")
    # ax_scores.set_ylabel("Score")

#
# 
#  From here: code for plotting the results
#
#

def graph_env_im(wbfe, wbfim, title_string = "{label}", ax_env_tylcv = None, ax_im_tylcv = None, ax_env_ccr = None, ax_im_ccr = None, ax_env_soil = None, ax_im_soil = None, ax_unc_tylcv = None, ax_unc_ccr = None, ax_unc_soil = None, cmap="gray"):
    """A generic function for plotting environments and their approximations into specific ax values. If an ax value is None, it will not plot."""
    # visualize the environment for tylcv
    if ax_env_tylcv != None:
        image_env_tylcv = ax_env_tylcv.imshow(wbfe.tylcv.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "TYLCV Env."
        evalstring = f"f'{title_string}'"
        ax_env_tylcv.set_title(eval(evalstring))
    # visualize the information model for tylcv
    if ax_im_tylcv != None:
        image_im_tylcv = ax_im_tylcv.imshow(wbfim.im_tylcv.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "TYLCV Estimate"
        evalstring = f"f'{title_string}'"
        ax_im_tylcv.set_title(eval(evalstring))

    # visualize the uncertainty of the information model for tylcv
    if ax_unc_tylcv != None:
        image_unc_tylcv = ax_unc_tylcv.imshow(wbfim.im_tylcv.uncertainty.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "TYLCV Uncertainty"
        evalstring = f"f'{title_string}'"
        ax_unc_tylcv.set_title(eval(evalstring))

    # visualize the environment for ccr
    if ax_env_ccr != None:
        image_env_ccr = ax_env_ccr.imshow(wbfe.ccr.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "CCR Env."
        evalstring = f"f'{title_string}'"
        ax_env_ccr.set_title(eval(evalstring))

    # visualize the information model for ccr
    if ax_im_ccr != None:
        image_im_ccr = ax_im_ccr.imshow(wbfim.im_ccr.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "CCR Estimate"
        evalstring = f"f'{title_string}'"
        ax_im_ccr.set_title(eval(evalstring))

    # visualize the uncertainty model for ccr
    if ax_unc_ccr != None:
        image_unc_ccr = ax_unc_ccr.imshow(wbfim.im_ccr.uncertainty.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "CCR Uncertainty"
        evalstring = f"f'{title_string}'"
        ax_unc_ccr.set_title(eval(evalstring))


    # visualize the environment for soil humidity
    if ax_env_soil != None:
        image_env_soil = ax_env_soil.imshow(wbfe.soil.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "Soil Humidity Env."
        evalstring = f"f'{title_string}'"
        ax_env_soil.set_title(eval(evalstring))

    # visualize the information model for soil humidity
    if ax_im_soil != None:
        image_im_soil = ax_im_soil.imshow(wbfim.im_soil.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "Soil Humidity Est."
        evalstring = f"f'{title_string}'"
        ax_im_soil.set_title(eval(evalstring))

    # visualize the uncertainty model for soil humidity
    if ax_unc_soil != None:
        image_unc_soil = ax_unc_soil.imshow(wbfim.im_soil.uncertainty.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
        label = "Soil Humidity Unc."
        evalstring = f"f'{title_string}'"
        ax_unc_soil.set_title(eval(evalstring))

def add_robot_path(results, ax, draw_it = True, pathcolor="blue", robotcolor = "green", draw_robot = True):
    """Adds the path of the robot to the figure (or not if )"""
    if not draw_it:
        return
    obsx = []
    obsy = []
    for obs in results["observations"]:
        obsx.append(obs[StoredObservationIM.X])
        obsy.append(obs[StoredObservationIM.Y])
    ax.add_line(lines.Line2D(obsx, obsy, color = pathcolor))
    if draw_robot:
        ax.add_patch(patches.Circle((results["robot"].x, results["robot"].y), radius=1, facecolor=robotcolor))


def end_of_day_graphs(results, graphfilename = "EndOfDayGraph.pdf", title = None, plot_uncertainty = True, ground_truth = "est+gt", score = None):
    """From the results of a 1 day experiment, create a figure that shows the
    environment, the information model at the end of the scenario, the path of the robot and the evolution of the score
    Ground truth = "est+gt", "est" or "gt" 
    """
    #print(results)
    wbfe = results["wbfe"]
    wbfim = results["estimator-code"]

    if ground_truth == "est+gt": # estimate and ground truth inline
        if plot_uncertainty:
            fig, ((ax_robot_path, ax_env_tylcv, ax_im_tylcv, ax_unc_tylcv, ax_env_ccr, ax_im_ccr, ax_unc_ccr, ax_env_soil, ax_im_soil, ax_unc_soil, ax_scores)) = plt.subplots(1, 11, figsize=(24,3))
        else:
            fig, ((ax_robot_path, ax_env_tylcv, ax_im_tylcv, ax_env_ccr, ax_im_ccr, ax_env_soil, ax_im_soil, ax_scores)) = plt.subplots(1, 8, figsize=(18,3))
    elif ground_truth == "est": # only estimate
        if plot_uncertainty:
            fig, ((ax_robot_path, ax_im_tylcv, ax_unc_tylcv, ax_im_ccr, ax_unc_ccr, ax_im_soil, ax_unc_soil, ax_scores)) = plt.subplots(1, 8, figsize=(18,3))
        else:
            fig, ((ax_robot_path, ax_im_tylcv, ax_im_ccr, ax_im_soil, ax_scores)) = plt.subplots(1, 5, figsize=(15,3))
    elif ground_truth == "gt": # only environment
        if plot_uncertainty:
            fig, ((ax_empty1, ax_env_tylcv, ax_empty2, ax_env_ccr, ax_empty3, ax_env_soil, ax_empty4, ax_empty5)) = plt.subplots(1, 8, figsize=(18,3))
            ax_empty1.axis('off')
            ax_empty2.axis('off')
            ax_empty3.axis('off')
            ax_empty4.axis('off')
            ax_empty5.axis('off')
        else:
            fig, ((ax_empty1, ax_env_tylcv, ax_env_ccr, ax_env_soil, ax_empty2)) = plt.subplots(1, 5, figsize=(15,3))
            ax_empty1.axis('off')
            ax_empty2.axis('off')
    else:
        raise f"Ground truth value {ground_truth} not understood"

    if title is None:
        fig.suptitle(f"{results['policy-name']}-{results['estimator-name']}", fontsize=16)
    elif title != "":
        fig.suptitle(title, fontsize=16)

    # visualize the observations, which gives us the path of the robot
    if ground_truth == "est+gt" or ground_truth == "est":
        empty = np.ones_like(wbfe.tylcv.value.T)
        image_env_tylcv = ax_robot_path.imshow(empty, vmin=0, vmax=1, origin="lower", cmap="gray")    
        ax_robot_path.set_title("Robot path")
        add_robot_path(results, ax_robot_path, draw_robot = False)
        graph_scores(results, ax_scores, score)

    if ground_truth == "est+gt": # estimate and ground truth inline
        if plot_uncertainty:
            graph_env_im(wbfe, wbfim, ax_env_tylcv=ax_env_tylcv, ax_im_tylcv=ax_im_tylcv, ax_unc_tylcv = ax_unc_tylcv, ax_env_ccr=ax_env_ccr, ax_im_ccr=ax_im_ccr, ax_unc_ccr = ax_unc_ccr, ax_env_soil=ax_env_soil, ax_im_soil=ax_im_soil, ax_unc_soil = ax_unc_soil, title_string="{label}")
        else:
            graph_env_im(wbfe, wbfim, ax_env_tylcv=ax_env_tylcv, ax_im_tylcv=ax_im_tylcv, ax_env_ccr=ax_env_ccr, ax_im_ccr=ax_im_ccr, ax_env_soil=ax_env_soil, ax_im_soil=ax_im_soil, title_string="{label}")
    elif ground_truth == "est": # only estimate
        if plot_uncertainty:
            graph_env_im(wbfe, wbfim, ax_im_tylcv=ax_im_tylcv, ax_unc_tylcv = ax_unc_tylcv, ax_im_ccr=ax_im_ccr, ax_unc_ccr = ax_unc_ccr, ax_im_soil=ax_im_soil, ax_unc_soil = ax_unc_soil, title_string="{label}")
        else:
            graph_env_im(wbfe, wbfim, ax_im_tylcv=ax_im_tylcv, ax_im_ccr=ax_im_ccr, ax_im_soil=ax_im_soil, title_string="{label}")
    elif ground_truth == "gt": # only ground truth
        if plot_uncertainty:
            graph_env_im(wbfe, wbfim, ax_env_tylcv=ax_env_tylcv,  ax_env_ccr=ax_env_ccr, ax_env_soil=ax_env_soil, title_string="{label}")
        else:
            graph_env_im(wbfe, wbfim, ax_env_tylcv=ax_env_tylcv, ax_env_ccr=ax_env_ccr, ax_env_soil=ax_env_soil, ax_im_soil=ax_im_soil, title_string="{label}")

    plt.savefig(pathlib.Path(results["results-basedir"], graphfilename))


def end_of_day_scores(results, graphfilename = "EndOfDayGraph.pdf", title = None):
    """
    Plot all the score components.
    """
    #print(results)
    wbfe = results["wbfe"]
    wbfim = results["estimator-code"]
    scores = WBF_MultiScore.score_components()
    fig, axes = plt.subplots(1, len(scores), figsize=(3*len(scores),3))
    for i, scorename in enumerate(scores):
        graph_scores(results, axes[i], scorename)
    plt.savefig(pathlib.Path(results["results-basedir"], graphfilename))


def hook_create_pictures(results, figsize = (3,3), draw_robot_path = True):
    """Hook for after day which generates the pictures of the graphs, for instance, for a movie"""

    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    path = results["results-path"]
    pnew = pathlib.Path(path.parent, "dir_" + path.name[4:])
    pnew.mkdir(exist_ok = True)
    results["picture-path"] = pnew
    logging.info(f"hook_create_pictures after {len(results['observations']):05d} observations")

    # tyclv-im-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_im_tylcv=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"tylcv-im-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # ccr-im-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_im_ccr=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"ccr-im-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # soil-im-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_im_soil=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"soil-im-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # tyclv-unc-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_unc_tylcv=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"tylcv-unc-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # ccr-unc-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_unc_ccr=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"ccr-unc-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # soil-unc-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_unc_soil=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"soil-unc-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # tyclv-env-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_env_tylcv=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"tylcv-env-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # ccr-env-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_env_ccr=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"ccr-env-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # soil-env-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_env_soil=ax)
    add_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"soil-env-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

