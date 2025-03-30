"""
wbf_simulate.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""

# import numpy as np
# import pathlib
import logging
#import pickle
#import copy
import time

#import bz2 as compress
# import gzip as compress

from policy import AbstractCommunicateAndFollowPath

# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

class TimeTrack:
    """Class for tracking time in the simulations"""
    def __init__(self):
        self.start_time = time.time_ns()
        self.last_start_time = self.start_time

    def policy_start(self):
        self.policy_start_time = time.time_ns()

    def policy_finish(self, results):
        self.policy_finish_time = time.time_ns()
        results["computation-cost-policy"].append(self.policy_finish_time - self.policy_start_time)

    def current(self, timestep, results):
        self.current_time = time.time_ns()
        if self.current_time - self.last_start_time > 10e9:
            print(f"At {timestep} / {int(results['timesteps-per-day'])} elapsed {int((self.current_time - self.start_time) / 1e9)} seconds")
            self.last_start_time = self.current_time


def simulate_1day(results):
    """
    Runs the simulation for one day. All the parameters and output is in the results dictionary.
    As this can also be slow, it performs time tracking, and marks the time tracking values into the results as well.
    """
    # we are assigning here to the robot the information model
    # FIXME: this might be more tricky later
    results["robot"].im = results["estimator-code"]
    results["scores"] = []
    results["observations"] = []
    results["positions"] = []
    results["computation-cost-policy"] = []
    results["time-track"] = TimeTrack()
    results["im_resolution_count"] = 0
    for timestep in range(int(results["timesteps-per-day"])):
        simulate_timestep_1robot(results, timestep)


def simulate_timestep_1robot(results, timestep):
    """Runs the simulation for 1 timestep with 1 robot"""
    results["time-track"].policy_start()
    results["robot"].enact_policy()
    results["robot"].proceed(1)
    position = [int(results["robot"].x), int(results["robot"].y), timestep]
    # print(results["robot"])
    results["positions"].append(position)
    obs = results["wbfe"].get_observation(position)
    results["observations"].append(obs)
    results["estimator-code"].add_observation(obs)
    results["robot"].add_observation(obs)
    results["time-track"].policy_finish(results)
    # update the im and the score at every im_resolution steps and at the last iteration
    results["im_resolution_count"] += 1
    if results["im_resolution_count"] == results["im_resolution"] or timestep + 1 == results["timesteps-per-day"]:
        results["estimator-code"].proceed(results["im_resolution"])
        results["score"] = results["score-code"].score(results["wbfe"], results["estimator-code"])
        for i in range(results["im_resolution_count"]):
            results["scores"].append(results["score"])
        results["im_resolution_count"] = 0
    if "hook-after-day" in results:
        results["hook-after-day"](results)
    results["time-track"].current(timestep, results)


def simulate_1day_multirobot(results):
    """
    Runs the simulation for one day. All the parameters and output is in the results dictionary.
    As this can also be slow, it performs time tracking, and marks the time tracking values into the results as well.
    """
    
    # we are assigning here to all the robots the information model
    # FIXME: this might be more tricky later
    for robot in results["robots"]:
        robot.im = results["estimator-code"]
    
    # positions, observations and scores are all list of lists
    # first index: time, second index: robot
    results["scores"] = []
    results["observations"] = []
    results["positions"] = []
    results["computation-cost-policy"] = []
    results["time-track"] = TimeTrack()

    results["im_resolution_count"] = 0 # counting the steps for the im update

    for timestep in range(int(results["timesteps-per-day"])):
        simulate_timestep_multirobot(results, timestep)


def simulate_timestep_multirobot(results, timestep):
    """Simulate one timestep in a multi-robot setting"""
    results["time-track"].policy_start()

    positions = []
    observations = []

    # communication rounds
    for round in range(results["communication-rounds"]):        
        for robot in results["robots"]:
            if isinstance(robot.policy, AbstractCommunicateAndFollowPath):
                robot.policy.act_send(round)
        for robot in results["robots"]:
            if isinstance(robot.policy, AbstractCommunicateAndFollowPath):
                msgs = results["communication"].receive(robot)
                robot.policy.act_receive(round, msgs)

    for robot in results["robots"]:
        robot.enact_policy()
        robot.proceed(1)
        position = [int(robot.x), int(robot.y), timestep]
        obs = results["wbfe"].get_observation(position)
        results["estimator-code"].add_observation(obs)
        robot.add_observation(obs)
        positions.append(position)
        observations.append(obs)

    results["positions"].append(positions)
    results["observations"].append(observations)

    results["time-track"].policy_finish(results)
    # update the im and the score at every im_resolution steps and at the last iteration
    results["im_resolution_count"] += 1
    if results["im_resolution_count"] == results["im_resolution"] or timestep + 1 == results["timesteps-per-day"]:
        results["estimator-code"].proceed(results["im_resolution"])
        results["score"] = results["score-code"].score(results["wbfe"], results["estimator-code"])
        for i in range(results["im_resolution_count"]):
            results["scores"].append(results["score"])
        results["im_resolution_count"] = 0
    if "hook-after-day" in results:
        results["hook-after-day"](results)
    results["time-track"].current(timestep, results)

