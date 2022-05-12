# allow the import from the source directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import pathlib
import logging
import pickle
import matplotlib.pyplot as plt

from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy, InformationGreedyPolicy
from PathGenerators import generate_lawnmower
from WaterberryFarm import create_wbfe, WaterberryFarm, MiniberryFarm, WaterberryFarmInformationModel, waterberry_score
from WbfExperiment import menu


logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

p = pathlib.Path.cwd()
figdir = pathlib.Path(p.parent, "__Temporary", p.name + "_data", "BenchmarkPaper")
figdir.mkdir(parents=True, exist_ok = True)

#
# this file runs the experiments and generates the figures for the paper
# "Waterberry Farms: a realistic benchmark for multi-robot informative path planning"
# 

def generate_env_pictures():
    """Generating the figures for the visualization of the environments. It will ask for the timepoint 4 times. After each visualization, close the visualization window!"""
    print(generate_env_pictures.__doc__)
    results = menu({"geometry": 1, "action": 2})
    figure = results["wbfe"].fig
    figure.savefig(pathlib.Path(figdir, "Env-Miniberry-10.jpg"))
    figure.savefig(pathlib.Path(figdir, "Env-Miniberry-10.pdf"))
    wbfe = results["wbfe"]
    fig, ax_geom = plt.subplots(1)
    wbfe.geometry.visualize(ax_geom)
    fig.savefig(pathlib.Path(figdir, "Geometry-Miniberry-10.jpg"))
    fig.savefig(pathlib.Path(figdir, "Geometry-Miniberry-10.pdf"))

    results = menu({"geometry": 2, "action": 2})
    figure = results["wbfe"].fig
    figure.savefig(pathlib.Path(figdir, "Env-Miniberry-30.jpg"))
    figure.savefig(pathlib.Path(figdir, "Env-Miniberry-30.pdf"))
    wbfe = results["wbfe"]
    fig, ax_geom = plt.subplots(1)
    wbfe.geometry.visualize(ax_geom)
    fig.savefig(pathlib.Path(figdir, "Geometry-Miniberry-30.jpg"))
    fig.savefig(pathlib.Path(figdir, "Geometry-Miniberry-30.pdf"))

    results = menu({"geometry": 3, "action": 2})
    figure = results["wbfe"].fig
    figure.savefig(pathlib.Path(figdir, "Env-Miniberry-100.jpg"))
    figure.savefig(pathlib.Path(figdir, "Env-Miniberry-100.pdf"))
    wbfe = results["wbfe"]
    fig, ax_geom = plt.subplots(1)
    wbfe.geometry.visualize(ax_geom)
    fig.savefig(pathlib.Path(figdir, "Geometry-Miniberry-100.jpg"))
    fig.savefig(pathlib.Path(figdir, "Geometry-Miniberry-100.pdf"))

    results = menu({"geometry": 4, "action": 2})
    print(list(results.keys()))
    figure = results["wbfe"].fig
    figure.savefig(pathlib.Path(figdir, "Env-Waterberry.jpg"))
    figure.savefig(pathlib.Path(figdir, "Env-Waterberry.pdf"))
    wbfe = results["wbfe"]
    fig, ax_geom = plt.subplots(1)
    wbfe.geometry.visualize(ax_geom)
    fig.savefig(pathlib.Path(figdir, "Geometry-Waterberry.jpg"))
    fig.savefig(pathlib.Path(figdir, "Geometry-Waterberry.pdf"))

def generate_movement_graph():
    """A single day single topic movement. Plot the environment and the information model in grayscale. Draw the robot path over the environment."""
    results = menu({"geometry": 1})
    print(results)
    fig, (ax_env, ax_im) = plt.subplots(1)



if __name__ == "__main__":
    generate_env_pictures()
