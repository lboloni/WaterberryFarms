"""
mrmr_graphics.py

Helper functions for creating the graphics for the MRMR paper
"""

from exp_run_config import Config
Config.PROJECTNAME = "WaterBerryFarms"

import pathlib
import matplotlib
import matplotlib.pyplot as plt
import wbf_figures
import logging
import numpy as np
import gzip as compress
import pickle
import pprint
import tqdm


from information_model import StoredObservationIM

logging.getLogger("fontTools").setLevel(logging.WARNING)


def load_back_results(experiment, listruns):
    """Loads back all the results of the experiment runs specified into a list"""
    all_results = {}

    for run in tqdm.tqdm(listruns):
        exp = Config().get_experiment(experiment, run)
        # pprint.pprint(exp)

        resultsfile = pathlib.Path(exp["data_dir"], "results.pickle")
        if not resultsfile.exists():
            print(f"Results file does not exist:\n{resultsfile}")
            print("Run the notebook Run-1Robot1Day with the same exp/run to create it.")
            raise Exception("Nothing to do.")

        # load the results file
        with compress.open(resultsfile, "rb") as f:
            results = pickle.load(f)    
        all_results[run] = results
    return all_results

def show_robot_with_plan(expall, scenario, results, robotno, t):
    """Visualize the plan of the robot at a certain time point"""

    ROBOT_COLORS = ["#E69F00", "#56B4E9", "#009E73"]
    robot_color = ROBOT_COLORS[2]

    robot = results["robots"][robotno]
    observations = [o[robotno] for o in results["observations"]]
    observations = observations[0:int(t)]
    if observations:
        print(f"Last observations: {observations[-1]}")

    oldplan = robot.oldplans[t]


    filename = f"plans_{scenario}_{robot.name}_{t}"

    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    wbf_figures.show_env_tylcv(results, ax)
    # obs = observations[1:int(t)]
    # color = "blue"
    wbf_figures.show_individual_robot_path(results, ax, robot=robot, observations=observations, pathcolor=robot_color, pathwidth=1,  draw_robot=False, robotcolor=robot_color, from_obs=0, to_obs=int(t))

    # add the plan
    # print(f"Oldplan beginning: {oldplan[0]}")
    planx = [a["x"] for a in oldplan]
    plany = [a["y"] for a in oldplan]
    ax.add_line(matplotlib.lines.Line2D(planx, plany, color = robot_color, linestyle=":", linewidth=1))

    # visualize the position of the robot
    #ax.add_patch(matplotlib.patches.Circle((observations[int(t)]["x"], observations[int(t)]["y"]), radius=3, facecolor=robot_color))
    if observations:
        ax.add_patch(matplotlib.patches.Circle((observations[-1]["x"], observations[-1]["y"]), radius=3, facecolor=robot_color))
    # ax.add_patch(matplotlib.patches.Circle((oldplan[0]["x"], oldplan[0]["y"]), radius=3, facecolor="yellow"))

    ax.set_title(f"{robot.name} at t={int(t)}")
    filepath = pathlib.Path(expall.data_dir(), f"{filename}.pdf")
    plt.savefig(filepath)
    print(f"Done saving to {filepath}")

def show_robot_trajectories_and_detections(exp_dest, name, results, robot_colors, lookup):
    """ Visualize detection paths for all running scenarios. Create a graph for the visualization of the paths, with the visualization of the detections
    exp_dest: the exprun whose data dir the figures are going to be put
    """
    fig_file = pathlib.Path(exp_dest.data_dir(), f"detections-map-{name}.pdf")
    if fig_file.exists():
        print(f"{fig_file} exists, skipping.")
        return
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    wbf_figures.show_env_tylcv(results, ax)
    if lookup and name in lookup:
        ax.set_title(lookup[name])
    else:
        print(f"Missing name map:\n{name}")
        ax.set_title(name)
    custom_lines = []
    labels = []

    for i, robot in enumerate(results["robots"]):
        color = robot_colors[i % len(results["robots"])]
        observations = [o[i] for o in results["observations"]]
        wbf_figures.show_individual_robot_path(results, ax, robot=robot, observations=observations, pathcolor=color, draw_robot=False)
        wbf_figures.show_individual_robot_detections(results, ax, robotno=i, detection_color=color, radius=0.5)
        # adding to the legend
        custom_lines.append(matplotlib.lines.Line2D([0], [0], color=color, lw=2))
        labels.append(robot.name)

    # Add both automatic and manual entries to the legend
    ax.legend(handles=[*custom_lines], labels=labels, ncol=3, bbox_to_anchor=(0.5, -0.1), loc="upper center", fontsize="9", columnspacing=0.5, labelspacing=0.5)
    #bbox_to_anchor=(0.5, 0))
             #, loc="upper center")

    # fig.legend(handles, labels, ncol=len(exps)+1,
    #        bbox_to_anchor=(0.5, 0), loc="upper center")

    plt.savefig(fig_file)
    plt.close()

def count_detections(results, robotno, field = "TYLCV"):
    """Returns the number of detections for the specified robot, adapted from wbf_figures.show_detections"""
    obs = np.array(results["observations"])[:, robotno]
    detections = [[a[StoredObservationIM.X], a[StoredObservationIM.Y]] for a in obs if a[field][StoredObservationIM.VALUE] == 0.0]
    return len(detections)

def show_agentwise_detections(exp_dest, name, results, robot_colors): 
    """create a bargraph with the number of detection points for each agent and a total for all agents in the specific run
    exp_dest: the exprun whose data dir the figures are going to be put
    """   
    fig_file = pathlib.Path(exp_dest.data_dir(), f"detections-bar-{name}.pdf")
    if fig_file.exists():
        print(f"{fig_file} exists, skipping.")
        return
    _, ax = plt.subplots(1,1, figsize=(3, 1.4))
    # ax.set_title(lookup[name])
    total = 0
    if "unclustered" in name:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 800)
    for i, robot in enumerate(results["robots"]):
        detections = count_detections(results, i)
        total += detections
        br = ax.bar(robot.name, detections, color=robot_colors[i])
    ax.bar("Total", total, color="gray")
    plt.savefig(fig_file)
    plt.close()

def show_comparative_detections(exp_dest, filename, values, lookup, name_colors):    
    """Create a comparative bargraph between the results listed in the values"""
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    ax.set_ylim(0, 600)
    for i, policyname in enumerate(values):
        if policyname in lookup:
            name = lookup[policyname]
        else: 
            print(f"No short name for:\n{policyname}")
            name = policyname
        br = ax.bar(name, values[policyname], color=name_colors[i%len
        (name_colors)])
    # rotate the labels, as they don't fit
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.savefig(pathlib.Path(exp_dest.data_dir(), f"comparative-bar-{filename}.pdf"))