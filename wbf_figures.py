"""
wbf_figures.py

Functions helping to plot the results of experiments performed with the Waterberry Farms benchmark
"""

from environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from information_model import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from robot import Robot
from policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy
from path_generators import find_fixed_budget_spiral, generate_lawnmower, find_fixed_budget_lawnmower, generate_spiral_path, find_fixed_budget_spiral
from water_berry_farm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, WBF_MultiScore

import numpy as np
import pathlib
import logging
import pickle
import copy
import time
import gzip as compress

from typing import List

import matplotlib
# configuring the fonts such that no Type 3 fonts are used
# requirement for ICRA
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["figure.autolayout"] = True

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import animation
import matplotlib.lines as lines

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

ROBOT_COLORS = ["red", "blue", "green", "orange", "pink"]


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



def end_of_day_graphs(results, graphfilename = "EndOfDayGraph.pdf", title = None, plot_uncertainty = True, ground_truth = "est+gt", score = None):
    """From the results of a 1 day experiment, create a figure that shows the
    environment, the information model at the end of the scenario, the path of the robot and the evolution of the score
    Ground truth = "est+gt", "est" or "gt" 
    """
    #print(results)
    wbfe = results["wbfe"]
    wbfim = results["estimator-CODE"]

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
        show_robot_path(results, ax_robot_path, draw_robot = False)
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
    wbfim = results["estimator-CODE"]
    scores = WBF_MultiScore.score_components()
    fig, axes = plt.subplots(1, len(scores), figsize=(3*len(scores),3))
    for i, scorename in enumerate(scores):
        graph_scores(results, axes[i], scorename)
    plt.savefig(pathlib.Path(results["results-basedir"], graphfilename))


def hook_create_pictures(results, figsize = (3,3), draw_robot_path = True):
    """Hook for after day which generates the pictures of the graphs, for instance, for a movie. This works for the single robot setting."""

    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-CODE"]
    path = results["results-path"]
    pnew = pathlib.Path(path.parent, "dir_" + path.name[4:])
    pnew.mkdir(exist_ok = True)
    results["picture-path"] = pnew
    logging.info(f"hook_create_pictures after {len(results['observations']):05d} observations")

    # tyclv-im-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_im_tylcv=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"tylcv-im-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # ccr-im-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_im_ccr=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"ccr-im-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # soil-im-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_im_soil=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"soil-im-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # tyclv-unc-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_unc_tylcv=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"tylcv-unc-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # ccr-unc-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_unc_ccr=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"ccr-unc-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # soil-unc-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_unc_soil=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"soil-unc-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # tyclv-env-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_env_tylcv=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"tylcv-env-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # ccr-env-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_env_ccr=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"ccr-env-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)

    # soil-env-robot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    graph_env_im(wbfe, wbfim, ax_env_soil=ax)
    show_robot_path(results, ax, draw_it = draw_robot_path)
    picname = f"soil-env-robot-{len(results['observations']):05d}.jpg"
    plt.savefig(pathlib.Path(results["results-basedir"], pathlib.Path(pnew, picname)))
    plt.close(fig)


def graph_gt_and_results(allresults, labels, metric="tylcv", horizontal=False):
    """Create a figure that shows the ground truth in the first column and the results of the algorithms, whose labels are listed in labels.
    The figure is saved in the directory from which the results were showed, 
    in a file that has a name formed from the concatenations of the labels"""
    uncmap = "inferno" # was "grey", "viridis"

    if horizontal:
        fig, axes = plt.subplots(nrows=3, ncols=(1+len(labels)), figsize=(3*(1+len(labels)),  3*3))
        #b = [ a[i][0] for i in range(len(a))]
        firstax = [ axes[i][0] for i in range(len(axes))]
    else:    
        fig, axes = plt.subplots(nrows=(1+len(labels)), ncols=3, figsize=(3*3,3*(1+len(labels))))
        firstax = axes[0]
    # print(firstax)
    firstax[0].axis('off')
    firstax[2].axis('off')

    # graph_env_im(allresults[labels[0]]["wbfe"], allresults[labels[0]]["estimator-CODE"], ax_env_tylcv=axes[0][1])
    graph_env_im(allresults[labels[0]]["wbfe"], allresults[labels[0]]["estimator-CODE"], **{f"ax_env_{metric}":firstax[1]})
    axes[1][0].set_title(f"Ground truth {metric}")
    filename = f"gt-{metric}-"
    for i, label in enumerate(labels):        
        results = allresults[label]
        filename += label + "-"
        if horizontal:
            axrow = [ axes[j][i+1] for j in range(len(axes))]
            #axrow = axes[:][i+1]
        else:    
            axrow = axes[i+1]
        # empty = np.ones_like(results["wbfe"].tylcv.value.T)
        empty = np.ones_like(vars(results["wbfe"])[metric].value.T)
        image_env = axrow[0].imshow(empty, vmin=0, vmax=1, origin="lower", cmap="gray")    
        show_robot_path(results, axrow[0], draw_robot = False)
        # graph_env_im(results["wbfe"], results["estimator-CODE"], ax_im_tylcv=axrow[1])
        graph_env_im(results["wbfe"], results["estimator-CODE"], **{f"ax_im_{metric}" : axrow[1]})
        # graph_env_im(results["wbfe"], results["estimator-CODE"], ax_unc_tylcv = axrow[2], cmap=uncmap)
        graph_env_im(results["wbfe"], results["estimator-CODE"], cmap=uncmap, **{f"ax_unc_{metric}": axrow[2]})
        axrow[0].set_title(f"{label} path")
        axrow[1].set_title(f"{label} estimate")
        axrow[2].set_title(f"{label} uncertainty")
    plt.tight_layout()
    filename = filename[:-1] + ".pdf"
    plt.savefig(pathlib.Path(results["results-basedir"], filename))  

def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    """ Exponential moving average used by Tensorboard """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed


def plot_scores(allresults, labels, scores, directory, smoothing = 0.7):
    """Taking the dictionary allresults, for all the specified labels, plot all the specified scored and save them into files in the directory"""

    for scorename in scores:
        fig, ax_scores = plt.subplots(1, figsize=(6,4))
        filename = f"score-{scorename}-sm{smoothing}"
        for label in labels:
            filename += label + "-"
            # rawscore = allresults[label]["scores"]
            results = allresults[label]
            scores = [a[scorename] for a in results["scores"]]
            # scores = smooth(scores, 0.99)
            scores = smooth(scores, 0.7)
            ax_scores.plot(scores, label = label)
        # ax_scores.set_ylim(top=2)
        ax_scores.set_xlabel("Time")
        ax_scores.set_ylabel("Score")
        ax_scores.set_title(f"Scores {scorename}")
        ax_scores.legend()
        fig.tight_layout()
        filename = pathlib.Path(directory, f"{filename[:-1]}.pdf")
        plt.savefig(filename)

def load_all_results(directory, prefix = "res_Miniberry-30_1M_"):
    """Loading all the results from a directory into a dictionary of results. The labels are inferred from the filename, with the common prefix stripped away"""
    allresults = {}
    for a in directory.iterdir():
        if a.name.startswith("res_") and "picgen" not in a.name:
            label = a.name[len(prefix):]
            # print(label)
            with compress.open(a, "rb") as f:
                results = pickle.load(f)
            allresults[label] = results
    return allresults

# *********************************************************
# 
#  From here: a new set of functions, operating on results
#  and hopefully easier to use
#
# *********************************************************


def show_empty(results, ax, title_string = "{label}"):
    """Create an empty graph of the appropriate size, 
    used for future drawing, eg. paths, plans or detections"""
    wbfe = results["wbfe"]
    empty = np.ones_like(wbfe.tylcv.value.T)
    image = ax.imshow(empty, vmin=0, vmax=1, origin="lower", cmap="gray")    
    label = "Robot path."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_env_tylcv(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the environment for tylcv"""
    wbfe = results["wbfe"]
    image = ax.imshow(wbfe.tylcv.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "TYLCV Env."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_im_tylcv(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the information model for tylcv"""
    wbfim = results["estimator-CODE"]
    image = ax.imshow(wbfim.im_tylcv.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "TYLCV IM."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_unc_tylcv(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the uncertainty of the information model for tylcv"""
    wbfim = results["estimator-CODE"]
    image = ax.imshow(wbfim.im_tylcv.uncertainty.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "TYLCV Uncertainty"
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_env_ccr(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the environment for ccr"""
    wbfe = results["wbfe"]
    image = ax.imshow(wbfe.ccr.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "CCR Env."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_im_ccr(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the information model for ccr"""
    wbfim = results["estimator-CODE"]
    image = ax.imshow(wbfim.im_ccr.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "CCR IM."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_unc_ccr(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the uncertainty of the information model for ccr"""
    wbfim = results["estimator-CODE"]
    image = ax.imshow(wbfim.im_ccr.uncertainty.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "CCR Uncertainty"
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_env_soil(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the environment for soil"""
    wbfe = results["wbfe"]
    image = ax.imshow(wbfe.soil.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "Soil Env."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_im_soil(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the information model for ccr"""
    wbfim = results["estimator-CODE"]
    image = ax.imshow(wbfim.im_soil.value.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "Soil IM."
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_unc_soil(results, ax, title_string = "{label}", cmap="gray"):
    """visualize the uncertainty of the information model for ccr"""
    wbfim = results["estimator-CODE"]
    image = ax.imshow(wbfim.im_soil.uncertainty.T, vmin=0, vmax=1, origin="lower", cmap=cmap)
    label = "Soil Uncertainty"
    evalstring = f"f'{title_string}'"
    ax.set_title(eval(evalstring))
    return image

def show_detections(results, ax, field = "TYLCV", detection_color="Blue", radius=0.5):
    """Shows the detections for all robots"""
    if "robots" not in results: # single robot
        show_individual_robot_detections(results, ax, -1, field, detection_color=detection_color, radius=radius)
    else: # multiple robots
        for i, robot in enumerate(results["robots"]):
            color = ROBOT_COLORS[i % len(results["robots"])]
            show_individual_robot_detections(results, ax, robotno=i, field=field, detection_color=color, radius=radius)

def show_individual_robot_detections(results, ax, robotno=1, field = "TYLCV", detection_color="Blue", radius=0.5):
    """Shows the detections for one specific robot. 
    """
    if robotno == -1:
        obs = np.array(results["observations"])[:]
    else:
        obs = np.array(results["observations"])[:, robotno]
    detections = [[a[StoredObservationIM.X], a[StoredObservationIM.Y]] for a in obs if a[field][StoredObservationIM.VALUE] == 0.0]
     
    for point in detections:
        ax.add_patch(matplotlib.patches.Circle((point[0], point[1]), radius=radius, facecolor=detection_color))


def show_robot_path(results, ax, draw_it = True, pathcolor="blue", 
                   pathwidth=1, robotcolor = "green", draw_robot = True, from_obs=-1, to_obs=-1, robot_number = -1):
    """
    Adds the path of a robot or robots to the figure, based on the observations.
    If robot_number is -1, means all the robots.
    """
    if not draw_it:
        return
    
    if robot_number == -1: # all robots
        if "robots" not in results: # single robot
            observations = results["observations"]
            robot = results["robot"]
            show_individual_robot_path(results, ax, robot, observations, pathcolor, pathwidth, robotcolor, draw_robot, from_obs, to_obs)
        else: # paint all robots
            for i, robot in enumerate(results["robots"]):
                color = ROBOT_COLORS[i % len(results["robots"])]
                observations = [o[i] for o in results["observations"]]
                show_individual_robot_path(results, ax, robot, observations, pathcolor = color, pathwidth = pathwidth, robotcolor = color, draw_robot=draw_robot, from_obs=from_obs, to_obs = to_obs)                             
    else: # we assume there are multiple robots and we want one
        observations = [o[robot_number] for o in results["observations"]]
        robot = results["robots"][robot_number]
        show_individual_robot_path(results, ax, robot, observations, pathcolor, pathwidth, robotcolor, draw_robot, from_obs, to_obs)

def show_individual_robot_path(results, ax, robot, observations, 
                              pathcolor="blue", pathwidth=1, robotcolor = "green", draw_robot = True, from_obs=-1, to_obs=-1):
    """Adds the path of an individual robot. Normally called from 
    show_robot_path"""
    obsx = []
    obsy = []
    if from_obs == -1:
        from_obs = 0
    if to_obs == -1:
        to_obs = len(observations)
    obses = observations[from_obs:to_obs]
    for obs in obses:
        obsx.append(obs[StoredObservationIM.X])
        obsy.append(obs[StoredObservationIM.Y])
    ax.add_line(lines.Line2D(obsx, obsy, color = pathcolor, linewidth=pathwidth))
    if draw_robot:
        ax.add_patch(patches.Circle((robot.x, robot.y), radius=3, facecolor=robotcolor))
