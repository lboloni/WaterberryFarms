"""
wbf_helper.py

Helper functions that are using the Experiment/Run configuration framework. Functions for the creation of environments etc. 

"""

from WaterberryFarm import WaterberryFarm, MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_DiskEstimator, WBF_IM_GaussianProcess, WBF_Score_WeightedAsymmetric
from Policy import RandomWaypointPolicy

import gzip as compress
import pickle
import pathlib


def create_wbf(exp):
    """Factory function for creating a WBF from an experiment 
    based on the typename"""
    if exp["typename"] == "Miniberry-10":
        return MiniberryFarm(scale=1)
    elif exp["typename"] == "Miniberry-30":
        return MiniberryFarm(scale=3)
    elif exp["typename"] == "Miniberry-100":
        return MiniberryFarm(scale=10)
    elif exp["typename"] == "Waterberry":
        return WaterberryFarm()
    else:
        raise Exception(f"Unknown type {exp['typename']}")


def create_wbfe(exp):
    """Helper function for the creation of a waterberry farm environment on which we can run experiments. It performs a caching process, if the files already exists, it just reloads them. This will save time for expensive simulations."""

    path_geometry = pathlib.Path(exp["data_dir"], "farm_geometry")
    path_environment = pathlib.Path(exp["data_dir"], "farm_environment")

    # caching: if it already exists, return it.
    if path_geometry.exists():
        print("loading the geometry and environment from saved data")
        with compress.open(path_geometry, "rb") as f:
            wbf = pickle.load(f)
        print("loading done")
        wbfe = WaterberryFarmEnvironment(wbf, use_saved=True, seed=10, savedir=exp["data_dir"])
        return wbf, wbfe
    
    # in this case, we assume that we need to create the whole thing
    wbf = create_wbf(exp)
    wbf.create_type_map()
    wbfe = WaterberryFarmEnvironment(wbf, use_saved=False, seed=10, savedir=exp["data_dir"])
    with compress.open(path_geometry, "wb") as f:
        pickle.dump(wbf, f)
    with compress.open(path_environment, "wb") as f:
        pickle.dump(wbfe, f)
    return wbf, wbfe

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


def create_policy(exp_policy, exp_env):
    """Create a policy, based on the specification of the policy and the environment. The name of the policy will be set according to the exp/run"""
    if exp_policy["policy-code"] == "RandomWaypointPolicy":
        geo = get_geometry(exp_env["typename"])
        # a random waypoint policy
        # geo = copy.copy(geom)
        policy = RandomWaypointPolicy(
            vel = 1, low_point = [geo["xmin"], 
            geo["ymin"]], high_point = [geo["xmax"], geo["ymax"]], seed = exp_policy["seed"])  
        policy.name = exp_policy["policy-name"]
        return policy
    raise Exception(f"Unsupported policy type {exp_policy['policy-code']}")

def create_estimator(exp_estimator, exp_env):
    """Create an estimator, based on the specification of the estimator and the environment. The name of the estimator will be set according to the exp/run""" 
    if exp_estimator["estimator-code"] == "WBF_IM_DiskEstimator":
        geo = get_geometry(exp_env["typename"])    
        estimator = WBF_IM_DiskEstimator(geo["width"], geo["height"])
        estimator.name = exp_estimator["estimator-name"]
        return estimator
    raise Exception(f"Unsupported estimator type {exp_policy['estimator-code']}")

def create_score(exp_score, exp_env):
    if exp_score["score-code"] == "WBF_Score_WeightedAsymmetric":
        score = WBF_Score_WeightedAsymmetric()
        score.name = exp_score["score-name"]
        return score
    raise Exception(f"Unsupported score type {exp_policy['score-code']}")