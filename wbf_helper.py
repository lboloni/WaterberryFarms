"""
wbf_helper.py

Helper functions that are using the Experiment/Run configuration framework. Functions for the creation of environments etc. 

"""

from water_berry_farm import WaterberryFarm, MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_DiskEstimator, WBF_IM_GaussianProcess, WBF_Score_WeightedAsymmetric
from policy import RandomWaypointPolicy, FollowPathPolicy
from path_generators import find_fixed_budget_lawnmower
from papers.y2025_mrmr.mrmr_policies import MRMR_Pioneer, MRMR_Contractor

import gzip as compress
import pickle
import pathlib
import imageio.v2 as imageio
import matplotlib.pyplot as plt

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
    """Returns an object with the geometry for the different types (or adds it into the passed dictionary). 
    FIXME: It calculates a specific timesteps per day for each size. I think that this was used to calculate the fixed budget lawnmower, but it is not appropriate to do it here!"""
    if geo == None:
        geo = {}
    geo["velocity"] = 1

    if typename == "Miniberry-10":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 0, 10, 0, 10
        geo["width"], geo["height"] = 11, 11
        #geo["timesteps-per-day"] = 0.4 * 100 
    elif typename == "Miniberry-30":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 0, 30, 0, 30
        geo["width"], geo["height"] = 31, 31
        #geo["timesteps-per-day"] = 0.4 * 900
    elif typename == "Miniberry-100":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 0, 100, 0, 100
        geo["width"], geo["height"] = 101, 101
        #geo["timesteps-per-day"] = 0.4 * 10000
    elif typename == "Waterberry":
        geo["xmin"], geo[
            "xmax"], geo["ymin"], geo["ymax"] = 1000, 5000, 1000, 4000
        geo["width"], geo["height"] = 5001, 4001
        # geo["timesteps-per-day"] = 0.4 * 12000000
    return geo


def create_policy(exp_policy, exp_env):
    """Create a policy, based on the specification of the policy and the environment. The name of the policy will be set according to the exp/run"""

    #
    #  Random waypoint policy
    #
    if exp_policy["policy-code"] == "RandomWaypointPolicy":
        geo = get_geometry(exp_env["typename"])
        policy = RandomWaypointPolicy(
            vel = 1, low_point = [geo["xmin"], 
            geo["ymin"]], high_point = [geo["xmax"], geo["ymax"]], seed = exp_policy["seed"])  
        policy.name = exp_policy["policy-name"]
        return policy
    #
    #  Fixed budget lawnmover: takes the budget from 
    # 
    if exp_policy["policy-code"] == "FixedBudgetLawnMower":
        geo = get_geometry(exp_env["typename"])
        # FIXME: maybe here I can specify a percentage budget....
        budget = exp_policy["budget"]
        # check if the area was passed
        if "area" in exp_policy:
            corners = eval(exp_policy["area"]) # [xmin, ymin, xmax, ymax]
            path = find_fixed_budget_lawnmower([0,0], corners[0], corners[2], corners[1], corners[3], geo["velocity"], time = budget)
        else:
            path = find_fixed_budget_lawnmower([0,0], geo["xmin"], geo["xmax"], geo["ymin"], geo["ymax"], geo["velocity"], time = budget)
        policy = FollowPathPolicy(vel = geo["velocity"], waypoints = path, repeat = True)
        policy.name = exp_policy["policy-name"] 
        # "FixedBudgetLawnmower"
        return policy
    #
    # MRMR policies
    #
    if exp_policy["policy-code"] == "MRMR_Pioneer":
        return MRMR_Pioneer(exp_policy, exp_env)
    if exp_policy["policy-code"] == "MRMR_Contractor":
        return MRMR_Contractor(exp_policy, exp_env)
    

    raise Exception(f"Unsupported policy type {exp_policy['policy-code']}")

def create_estimator(exp_estimator, exp_env):
    """Create an estimator, based on the specification of the estimator and the environment. The name of the estimator will be set according to the exp/run""" 
    #
    #  Adaptive disk estimator
    # 
    if exp_estimator["estimator-code"] == "WBF_IM_DiskEstimator":
        geo = get_geometry(exp_env["typename"])    
        estimator = WBF_IM_DiskEstimator(geo["width"], geo["height"])
        estimator.name = exp_estimator["estimator-name"]
        return estimator
    #
    #  Gaussian process estimator
    #
    if exp_estimator["estimator-code"] == "WBF_IM_GaussianProcess":
        geo = get_geometry(exp_env["typename"])    
        estimator = WBF_IM_GaussianProcess(geo["width"], geo["height"])
        estimator.name = exp_estimator["estimator-name"]
        return estimator

    raise Exception(f"Unsupported estimator type {exp_estimator['estimator-code']}")

def create_score(exp_score, exp_env):
    if exp_score["score-code"] == "WBF_Score_WeightedAsymmetric":
        score = WBF_Score_WeightedAsymmetric()
        score.name = exp_score["score-name"]
        return score
    raise Exception(f"Unsupported score type {exp_score['score-code']}")

def create_wbfe_custom(exp_env):
    """Creates a custom WBFE for the TYLCV. It creates the specified png file
    if it does not exist. Use some image editor, such as GIMP to edit the 
    values. 
    FIXME: extend to the CCR and soil fields. This is experimental stuff. 
    """
    wbf, wbfe = create_wbfe(exp_env)
    # we have to "proceed" on it, but it doesn't matter how much
    #wbfe.proceed(exp["time-start-environment"])
    wbfe.proceed(1)
    # check if there is a custom field in the environment
    if "custom-tylcv" in exp_env:
        exp_filename = exp_env["exp_run_sys_indep_file"]
        exp_path = pathlib.Path(exp_filename).parent
        custom_env_file = pathlib.Path(exp_path, exp_env["custom-tylcv"])
        if custom_env_file.exists():
            print(f"loading from {custom_env_file}")
            loaded_array = imageio.imread(custom_env_file)
            print(loaded_array)
            if loaded_array.ndim == 3:
                one_channel = loaded_array[:, :, 0]  # 0=Red, 1=Green, 2=Blue
            else:
                one_channel = loaded_array  # already grayscale
            # overwrite the field
            wbfe.tylcv.value = one_channel
            # changes the environment to all tomato
            wbf.patches = []
            area = [[0,0], [wbfe.width,0], [wbfe.width, wbfe.height], [0, wbfe.height]]
            wbf.add_patch("all-tylcv", type="tomato", area = area, color="blue")
        else:
            print(f"custom env. file {custom_env_file} does not exist.")
            
            plt.imsave(custom_env_file, wbfe.tylcv.value, cmap='gray')
    else:
        print("this environment is not really custom")
    return wbf, wbfe