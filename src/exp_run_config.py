"""
exp_run_config.py

The Waterberry Farms settings of the exp/run framework, which is implemented
by the ExpRunFlow library (exprunflow.exp_run_config).
"""

import pathlib

from exprunflow.exp_run_config import Config, Experiment

Config.PROJECTNAME = "WaterBerryFarms"
Config.SRC_ROOT = pathlib.Path(__file__).resolve().parent
