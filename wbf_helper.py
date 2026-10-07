"""
wbf_helper.py

Helper functions that are using the Experiment/Run configuration framework. Functions for the creation of environments etc. 

"""

from exp_run_config import Config, Experiment
from environment import ScalarFieldEnvironment
from water_berry_farm import WaterberryFarm, MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_DiskEstimator, WBF_IM_GaussianProcess
from policy import FollowPathPolicy
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from path_generators import find_fixed_budget_lawnmower

import gzip as compress
import pickle
import pathlib
import yaml
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


def create_estimator(exp_estimator, geometry):
    """Factory function for creating a WBF estimator from an estimator exp"""
    if exp_estimator["estimator-type"] == "disk":
        estimator = WBF_IM_DiskEstimator(
            geometry["width"], geometry["height"],
            disk_radius=exp_estimator["disk-radius"],
            default_tylcv=exp_estimator["default-tylcv"],
            default_ccr=exp_estimator["default-ccr"],
            default_soil=exp_estimator["default-soil"])
    elif exp_estimator["estimator-type"] == "gaussian-process":
        kernel = RBF(length_scale=[exp_estimator["gp-length-scale"]] * 2,
                     length_scale_bounds=exp_estimator["gp-length-scale-bounds"]) \
            + WhiteKernel(noise_level=exp_estimator["gp-noise"])
        estimator = WBF_IM_GaussianProcess(
            geometry["width"], geometry["height"], gp_kernel=kernel,
            gp_restarts=exp_estimator["gp-restarts"],
            gp_normalize_y=exp_estimator["gp-normalize-y"],
            default_tylcv=exp_estimator["default-tylcv"],
            default_ccr=exp_estimator["default-ccr"],
            default_soil=exp_estimator["default-soil"])
    else:
        raise Exception(f"Unknown estimator type {exp_estimator['estimator-type']}")
    estimator.name = exp_estimator["estimator-name"]
    return estimator


def create_wbfe(exp):
    """Helper function for the creation of a waterberry farm environment on which we can run experiments. It performs a caching process, if the files already exists, it just reloads them. This will save time for expensive simulations."""

    path_geometry = pathlib.Path(exp["data_dir"], "farm_geometry")
    path_environment = pathlib.Path(exp["data_dir"], "farm_environment")

    # The cached geometry identifies a fully initialized standard or custom
    # environment. Subsequent runs replay its precalculated field values.
    if path_geometry.exists():
        print("loading the geometry and environment from saved data")
        with compress.open(path_geometry, "rb") as f:
            wbf = pickle.load(f)
        print("loading done")
        wbfe = WaterberryFarmEnvironment(wbf, use_saved=True, seed=10, savedir=exp["data_dir"])
        return wbf, wbfe
    
    # in this case, we assume that we need to create the whole thing
    wbf = create_wbf(exp)
    if "custom-tylcv" in exp:
        customize_geometry(wbf)
    wbf.create_type_map()
    wbfe = WaterberryFarmEnvironment(wbf, use_saved=False, seed=10, savedir=exp["data_dir"])
    if "custom-tylcv" in exp:
        customize_environment(wbfe, exp)
    with compress.open(path_geometry, "wb") as f:
        pickle.dump(wbf, f)
    with compress.open(path_environment, "wb") as f:
        pickle.dump(wbfe, f)
    return wbf, wbfe

def precompute_environment(run):
    """Precomputes the environment exp/run for its precompute-time, unless an earlier precomputation 
    completed (its exprun.yaml contains time_done). Returns the environment exp."""
    exp_env = Config().get_experiment("environment", run)
    with open(pathlib.Path(exp_env["data_dir"], "exprun.yaml")) as f:
        if Config.TIME_DONE in yaml.safe_load(f):
            return exp_env
    exp_env = Config().get_experiment("environment", run, creation_style="discard-old")
    wbf, wbfe = create_wbfe(exp_env)
    for _ in range(exp_env["precompute-time"]):
        wbfe.proceed()
    exp_env.done()
    return exp_env

def customize_geometry(wbf):
    """Make the existing farm area one tomato patch without changing its size."""
    width, height = wbf.width, wbf.height
    wbf.patches = []
    area = [
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1],
    ]
    wbf.add_patch("all-tylcv", type="tomato", area=area, color="blue")


def customize_environment(wbfe, exp_env):
    """Creates a custom WBFE for the TYLCV. It creates the specified png file
    if it does not exist. Use some image editor, such as GIMP to edit the 
    values. 
    FIXME: extend to the CCR and soil fields. This is experimental stuff. 
    """
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
        custom_value = one_channel / 255.0
        wbfe.tylcv.environment = ScalarFieldEnvironment(
            "TYLCV", wbfe.width, wbfe.height, seed=0, value=custom_value)
        wbfe.proceed(1)
    else:
        wbfe.proceed(1)
        print(f"custom env. file {custom_env_file} does not exist")
        plt.imsave(custom_env_file, wbfe.tylcv.value, cmap='gray')
    return wbfe

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


def generate_fixed_budget_lawnmower(exp_policy: Experiment,
                                    exp_env: Experiment):
    """Example of how to create a generator for a policy type, in this case fixed budget lawnmower FBLM"""
    geo = get_geometry(exp_env["typename"])
    # FIXME: maybe here I can specify a percentage budget....
    budget = exp_policy["budget"]
    # check if the area was passed
    if "area" in exp_policy:
        corners = exp_policy["area"] # [xmin, ymin, xmax, ymax]
        path = find_fixed_budget_lawnmower([0,0], corners[0], corners[2], corners[1], corners[3], geo["velocity"], time = budget)
    else:
        path = find_fixed_budget_lawnmower([0,0], geo["xmin"], geo["xmax"], geo["ymin"], geo["ymax"], geo["velocity"], time = budget)
    policy = FollowPathPolicy(vel = geo["velocity"], waypoints = path, repeat = True)
    policy.name = exp_policy["policy-name"] 
    # "FixedBudgetLawnmower"
    return policy
