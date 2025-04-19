"""
wbf_simulate.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""

import logging
import pickle
import time
import pathlib
import gzip as compress

#import bz2 as compress
# import gzip as compress

from policy import AbstractCommunicateAndFollowPath
from exp_run_config import Config
Config.PROJECTNAME = "WaterBerryFarms"
from pprint import pprint
import gzip as compress
from wbf_helper import get_geometry, create_wbfe, create_policy, create_estimator, create_score
from robot import Robot

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
    results["robot"].im = results["estimator-CODE"]
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
    results["estimator-CODE"].add_observation(obs)
    results["robot"].add_observation(obs)
    results["time-track"].policy_finish(results)
    # update the im and the score at every im_resolution steps and at the last iteration
    results["im_resolution_count"] += 1
    if results["im_resolution_count"] == results["im_resolution"] or timestep + 1 == results["timesteps-per-day"]:
        results["estimator-CODE"].proceed(results["im_resolution"])
        results["score"] = results["score-code"].score(results["wbfe"], results["estimator-CODE"])
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
        robot.im = results["estimator-CODE"]
    
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
        results["estimator-CODE"].add_observation(obs)
        robot.add_observation(obs)
        positions.append(position)
        observations.append(obs)

    results["positions"].append(positions)
    results["observations"].append(observations)

    results["time-track"].policy_finish(results)
    # update the im and the score at every im_resolution steps and at the last iteration
    results["im_resolution_count"] += 1
    if results["im_resolution_count"] == results["im_resolution"] or timestep + 1 == results["timesteps-per-day"]:
        results["estimator-CODE"].proceed(results["im_resolution"])
        results["score"] = results["score-code"].score(results["wbfe"], results["estimator-CODE"])
        for i in range(results["im_resolution_count"]):
            results["scores"].append(results["score"])
        results["im_resolution_count"] = 0
    if "hook-after-day" in results:
        results["hook-after-day"](results)
    results["time-track"].current(timestep, results)

def save_simulation_results(resultsfile, results):
    """Saves the results of the simulation to a compressed pickle
    file. 
    To allow for the loading under every circumstances, it removes 
    all fields that are named "xxx-code"
    Things such as the estimator, which would require more stuff for this, will be named for the time being estimator-CODE
    """

    results_nc = {}
    for a in results:
        if not a.endswith("-code"):
            results_nc[a]=results[a]
    print(f"Saving results to: {resultsfile}")
    with compress.open(resultsfile, "wb") as f:
        pickle.dump(results_nc, f)    

def run_1robot1day(exp):
    """Take an experiment of type 1robot1day, set up the results
    based on the description in it, which includes the policy description. 
    Then runs simulate1day, and saves it to the experiment. """

    resultsfile = pathlib.Path(exp.data_dir(), "results.pickle")
    if resultsfile.exists():
        print(f"Results file already exists:\n{resultsfile}")
        print(f"Delete this file if re-running is desired.")
        # raise Exception("Nothing to do.")
        return

    # the exp for the environment
    exp_env = Config().get_experiment("environment", exp["exp_environment"])
    pprint(exp_env)
    results = {}

    #
    # Setting the policy based on the exp for policy
    #
    exp_policy = Config().get_experiment("policy", exp["exp_policy"])
    pprint(exp_policy)
    if exp_policy["policy-code"] == "-":
        # the policy is created through a policy generator that is evaluated
        # this is for new code models
        generator = exp_policy["policy-code-generator"]        
        policy = eval(generator)(exp_policy, exp_env)
    else:
        policy = create_policy(exp_policy, exp_env)
    results["policy-code"] = policy
    results["policy-name"] = results["policy-code"].name
    #
    # End of setting the policy
    #

    #
    # Setting the estimator code based on the exp for estimator
    #
    exp_estimator = Config().get_experiment("estimator", exp["exp_estimator"])
    pprint(exp_estimator)
    results["estimator-CODE"] = create_estimator(exp_estimator, exp_env)
    results["estimator-name"] = results["estimator-CODE"].name
    #
    # End of setting the estimator
    #

    #
    # Setting the score code based on the exp for the score
    #
    exp_score = Config().get_experiment("score", exp["exp_score"])
    pprint(exp_score)
    results["score-code"] = create_score(exp_score, exp_env)
    results["score-name"] = results["score-code"].name
    #
    # End of setting the score 
    #
    results["velocity"] = exp["velocity"]
    results["timesteps-per-day"] = exp["timesteps-per-day"]
    results["time-start-environment"] = exp["time-start-environment"]
    results["im_resolution"] = exp["im_resolution"]
    results["results-basedir"] = exp["data_dir"]
    results["action"] = "run-one-day"
    results["typename"] = exp_env["typename"]
    wbf, wbfe = create_wbfe(exp_env)
    # move ahead to the starting point of the environment
    wbfe.proceed(results["time-start-environment"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["days"] = 1
    get_geometry(results["typename"], results)
    # create the robot and set the policy
    results["robot"] = Robot("Rob", 0, 0, 0, env=None, im=None)
    results["robot"].assign_policy(results["policy-code"])
    # 
    # This is where we actually calling the simulation
    #
    simulate_1day(results)
    #print(f"Saving results to: {resultsfile}")
    #with compress.open(resultsfile, "wb") as f:
    #    pickle.dump(results, f)
    save_simulation_results(resultsfile, results)
    exp.done()