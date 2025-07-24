"""
mrmr_graphics.py

Helper functions for creating the graphics for the MRMR paper
"""

import pathlib
import matplotlib
import matplotlib.pyplot as plt
import wbf_figures
import logging
import numpy as np
from information_model import StoredObservationIM

logging.getLogger("fontTools").setLevel(logging.WARNING)


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

    # position of the robot
    #ax.add_patch(matplotlib.patches.Circle((observations[int(t)]["x"], observations[int(t)]["y"]), radius=3, facecolor=robot_color))
    if observations:
        ax.add_patch(matplotlib.patches.Circle((observations[-1]["x"], observations[-1]["y"]), radius=3, facecolor=robot_color))
    # ax.add_patch(matplotlib.patches.Circle((oldplan[0]["x"], oldplan[0]["y"]), radius=3, facecolor="yellow"))

    ax.set_title(f"{robot.name} at t={int(t)}")
    filepath = pathlib.Path(expall.data_dir(), f"{filename}.pdf")
    plt.savefig(filepath)
    print(f"Done saving to {filepath}")

def show_robot_trajectories_and_detections(expall, name, results, robot_colors, lookup):
    """ Visualize detection paths for all running scenarios. Create a graph for the visualization of the paths, with the visualization of the detections
    """
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    wbf_figures.show_env_tylcv(results, ax)
    ax.set_title(lookup[name])
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

    plt.savefig(pathlib.Path(expall.data_dir(), f"detections-map-{name}.pdf"))

    plt.close()

def count_detections(results, robotno, field = "TYLCV"):
    """Returns the number of detections for the specified robot, adapted from wbf_figures.show_detections"""
    obs = np.array(results["observations"])[:, robotno]
    detections = [[a[StoredObservationIM.X], a[StoredObservationIM.Y]] for a in obs if a[field][StoredObservationIM.VALUE] == 0.0]
    return len(detections)

def show_agentwise_detections(expall, name, results, robot_colors): 
    """create a bargraph with the number of detection points for each of them"""   
    fig, ax = plt.subplots(1,1, figsize=(3, 1.4))
    # ax.set_title(lookup[name])
    total = 0
    if "unclustered" in name:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 600)
    for i, robot in enumerate(results["robots"]):
        detections = count_detections(results, i)
        total += detections
        br = ax.bar(robot.name, detections, color=robot_colors[i])
    ax.bar("Total", total, color="gray")
    plt.savefig(pathlib.Path(expall.data_dir(), f"detections-bar-{name}.pdf"))

def show_comparative_detections(expall, name, values, lookup, name_colors):    
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    ax.set_ylim(0, 600)
    for i, policyname in enumerate(values):
        br = ax.bar(lookup[policyname], values[policyname], color=name_colors[i])
    plt.savefig(pathlib.Path(expall.data_dir(), f"comparative-bar-{name}.pdf"))