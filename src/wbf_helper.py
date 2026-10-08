"""
wbf_helper.py

Helper functions that are using the Experiment/Run configuration framework. Functions for the creation of environments etc. 

"""

from exp_run_config import Config, Experiment
from environment import ScalarFieldEnvironment
from water_berry_farm import FarmGeometry, WaterberryFarm, MiniberryFarm, WaterberryFarmEnvironment, WBF_IM_Composite
from information_model import (PointEstimateScalarFieldIM, DiskEstimateScalarFieldIM, GaussianProcessScalarFieldIM,
    LocalGPScalarFieldIM, IndicatorGPScalarFieldIM, NearestScalarFieldIM, IDWScalarFieldIM, RBFScalarFieldIM,
    OccupancyGridIM, MRFScalarFieldIM)
from epidemic_filter import EpidemicParticleFilterIM
from policy import FollowPathPolicy
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from path_generators import find_fixed_budget_lawnmower

import gzip as compress
import hashlib
import pickle
import pathlib
import yaml

SRC_ROOT = pathlib.Path(__file__).resolve().parent
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


def create_field_estimator(estimator_type, exp_estimator, width, height, default_value):
    """Factory function for the scalar-field estimator of one field, from the parameters of an estimator exp"""
    e = exp_estimator
    if estimator_type == "point":
        return PointEstimateScalarFieldIM(width, height, default_value=default_value)
    if estimator_type == "disk":
        return DiskEstimateScalarFieldIM(width, height, disk_radius=e["disk-radius"], default_value=default_value)
    if estimator_type in ["gaussian-process", "gp-local", "gp-indicator"]:
        kernel = RBF(length_scale=[e["gp-length-scale"]] * 2, length_scale_bounds=e["gp-length-scale-bounds"]) \
            + WhiteKernel(noise_level=e["gp-noise"])
        gp = {"gp_kernel": kernel, "default_value": default_value, "n_restarts_optimizer": e["gp-restarts"],
              "normalize_y": e["gp-normalize-y"], "max_observations": e["gp-max-observations"]}
        if estimator_type == "gp-local":
            return LocalGPScalarFieldIM(width, height, tile_size=e["gp-tile-size"], overlap=e["gp-overlap"], **gp)
        if estimator_type == "gp-indicator":
            return IndicatorGPScalarFieldIM(width, height, **gp)
        return GaussianProcessScalarFieldIM(width, height, **gp)
    if estimator_type == "nearest":
        return NearestScalarFieldIM(width, height, default_value=default_value, length_scale=e["length-scale"])
    if estimator_type == "idw":
        return IDWScalarFieldIM(width, height, default_value=default_value, power=e["idw-power"],
                                k=e["idw-k"], length_scale=e["length-scale"])
    if estimator_type == "rbf":
        return RBFScalarFieldIM(width, height, default_value=default_value, kernel=e["rbf-kernel"],
            epsilon=e["rbf-epsilon"], smoothing=e["rbf-smoothing"], neighbors=e["rbf-neighbors"],
            length_scale=e["length-scale"])
    if estimator_type == "occupancy":
        return OccupancyGridIM(width, height, default_value=default_value, p_hit=e["occupancy-p-hit"],
            p_false_alarm=e["occupancy-p-false-alarm"], footprint=e["occupancy-footprint"], prior=e["disease-prior"])
    if estimator_type == "mrf":
        return MRFScalarFieldIM(width, height, default_value=default_value, coupling=e["mrf-coupling"],
                                prior=e["disease-prior"], iterations=e["mrf-iterations"])
    if estimator_type == "epidemic-pf":
        return EpidemicParticleFilterIM(width, height, default_value=default_value, particles=e["pf-particles"],
            p_transmission=e["pf-p-transmission"], infection_duration=e["pf-infection-duration"],
            infection_seeds=e["pf-infection-seeds"], spread_dimension=e["pf-spread-dimension"],
            days=e["pf-days"], localization_radius=e["pf-localization-radius"],
            observation_noise=e["pf-observation-noise"], seed=e["pf-seed"])
    if estimator_type == "cnn":
        # a paper-specific estimator with an optional dependency (PyTorch), imported only when used
        from papers.estimator_cnn.cnn_estimator import CNNScalarFieldIM
        return CNNScalarFieldIM(width, height, default_value=default_value,
                                model_path=pathlib.Path(SRC_ROOT, e["cnn-model-path"]))
    raise Exception(f"Unknown estimator type {estimator_type}")


def create_estimator(exp_estimator, geometry):
    """Factory function for creating a WBF estimator from an estimator exp. Every field has the estimator 
    type <field>-estimator-type, which defaults (null) to estimator-type."""
    fields = {}
    for name in ["tylcv", "ccr", "soil"]:
        estimator_type = exp_estimator[f"{name}-estimator-type"] or exp_estimator["estimator-type"]
        fields[name] = create_field_estimator(estimator_type, exp_estimator, geometry["width"],
                                              geometry["height"], exp_estimator[f"default-{name}"])
    return WBF_IM_Composite(geometry["width"], geometry["height"], fields["tylcv"], fields["ccr"],
                            fields["soil"], name=exp_estimator["estimator-name"])


FIELDS = ["tylcv", "ccr", "soil"]


def uses_cache(exp_env):
    """Whether the environment of the exp/run is cached. By default (cache: null) every environment
    is cached, except the full Waterberry farm, whose cache would be too large."""
    if exp_env["cache"] is None:
        return exp_env["typename"] != "Waterberry"
    return exp_env["cache"]


def field_parameters(exp_env):
    """The parameters of the field models of WaterberryFarmEnvironment, from the environment exp/run"""
    epidemic = lambda name: {
        "p_transmission": exp_env[f"{name}-p-transmission"],
        "infection_duration": exp_env[f"{name}-infection-duration"],
        "infection_seeds": exp_env[f"{name}-infection-seeds"],
        "spread_dimension": exp_env[f"{name}-spread-dimension"]}
    soil = {"seed": exp_env["soil-seed"], "evaporation": exp_env["soil-evaporation"],
            "rainfall": exp_env["soil-rainfall"], "rain_likelihood": exp_env["soil-rain-likelihood"]}
    return {"tylcv": epidemic("tylcv"), "ccr": epidemic("ccr"), "soil": soil}


def create_wbfe(exp):
    """Creates the waterberry farm geometry and environment of an environment exp/run. A cached 
    environment is replayed if its geometry was already saved, otherwise it is created and saved; 
    an environment without cache is created, and computed live, every time."""
    cache = uses_cache(exp)
    path_geometry = pathlib.Path(exp["data_dir"], "farm_geometry")
    path_environment = pathlib.Path(exp["data_dir"], "farm_environment")

    # The cached geometry identifies a fully initialized environment. 
    # Subsequent runs replay its precalculated field values.
    if cache and path_geometry.exists():
        print("loading the geometry and environment from saved data")
        with compress.open(path_geometry, "rb") as f:
            wbf = pickle.load(f)
        print("loading done")
        wbfe = WaterberryFarmEnvironment(wbf, use_saved=True, seed=exp["seed"], savedir=exp["data_dir"])
        return wbf, wbfe

    wbf = create_wbf(exp)
    wbf.replant(exp["planting"])
    wbf.create_type_map()
    wbfe = WaterberryFarmEnvironment(wbf, use_saved=False, seed=exp["seed"],
        savedir=exp["data_dir"] if cache else None, **field_parameters(exp))
    apply_pictures(wbfe, exp)
    if cache:
        with compress.open(path_geometry, "wb") as f:
            pickle.dump(wbf, f)
        with compress.open(path_environment, "wb") as f:
            pickle.dump(wbfe, f)
    return wbf, wbfe


def picture_path(exp_env, name):
    """The picture of a field, which is next to the exp/run file"""
    return pathlib.Path(pathlib.Path(exp_env["exp_run_sys_indep_file"]).parent, exp_env[f"{name}-picture"])


def apply_pictures(wbfe, exp_env):
    """Sets the fields that have a picture (<field>-picture) in the exp/run. The red (or gray) channel of 
    the picture, divided by 255, is indexed as [x, y]. In static mode, the picture is the value of the 
    field on every day; in initial mode (epidemics only), it is the initial status of the epidemic.
    If the picture does not exist, the model's field is written as a template to edit, and an exception 
    is raised."""
    for name in FIELDS:
        if exp_env[f"{name}-picture"] is None:
            continue
        path = picture_path(exp_env, name)
        field = getattr(wbfe, name)
        if not path.exists():
            model = field.environment
            value = model.value if name == "soil" else model.create_value()
            plt.imsave(path, value, cmap="gray", vmin=0, vmax=1)
            raise Exception(f"The picture {path} did not exist: a template was written there, edit it and create the environment again")
        loaded = imageio.imread(path)
        value = (loaded[:, :, 0] if loaded.ndim == 3 else loaded) / 255.0
        mode = "static" if name == "soil" else exp_env[f"{name}-picture-mode"]
        if mode == "static":
            field.environment = ScalarFieldEnvironment(field.environment.name, wbfe.width, wbfe.height, seed=0, value=value)
        elif mode == "initial":
            field.environment.set_initial_status(value)
        else:
            raise Exception(f"Unknown picture mode {mode}")


def environment_configuration(exp_env):
    """The values of the environment exp/run that determine its precomputed fields, including a 
    hash of the content of its existing pictures"""
    keys = ["typename", "planting", "precompute-time", "seed", "cache"]
    configuration = {key: value for key, value in exp_env.values.items() if key in keys or key.startswith(tuple(f"{name}-" for name in FIELDS))}
    configuration["geometry-version"] = FarmGeometry.RASTERIZATION_VERSION
    for name in FIELDS:
        if exp_env[f"{name}-picture"] is not None and picture_path(exp_env, name).exists():
            configuration[f"{name}-picture-sha1"] = hashlib.sha1(picture_path(exp_env, name).read_bytes()).hexdigest()
    return configuration


def cache_is_current(exp_env, saved):
    """Whether the saved exprun.yaml of the environment records a completed precomputation with the 
    same configuration as the environment exp/run"""
    return Config.TIME_DONE in saved and all(
        key in saved and saved[key] == value for key, value in environment_configuration(exp_env).items())


def precompute_environment(run):
    """Precomputes the environment exp/run for its precompute-time. Skipped if the environment has no 
    cache, or if an earlier precomputation completed (its exprun.yaml contains time_done) with the same 
    configuration. Returns the environment exp."""
    exp_env = Config().get_experiment("environment", run)
    if not uses_cache(exp_env):
        return exp_env
    with open(pathlib.Path(exp_env["data_dir"], "exprun.yaml")) as f:
        saved = yaml.safe_load(f)
    if cache_is_current(exp_env, saved):
        return exp_env
    exp_env = Config().get_experiment("environment", run, creation_style="discard-old")
    precompute(exp_env)
    return exp_env


def precompute(exp_env):
    """Evolves the cached environment of a freshly created exp/run for its precompute-time, records 
    its configuration in exprun.yaml and marks the exp/run done. An environment without cache is 
    only marked done, as there is nothing to precompute."""
    if uses_cache(exp_env):
        wbf, wbfe = create_wbfe(exp_env)
        for _ in range(exp_env["precompute-time"]):
            wbfe.proceed()
        exp_env.values.update(environment_configuration(exp_env))  # recorded with the picture hashes
    exp_env.done()


def get_geometry(typename, geo = None):
    """Returns the dimensions of the geometry type (or adds them into the passed dictionary): the grid size 
    (width, height), the owner's area as the inclusive range of its cells (xmin..xmax, ymin..ymax), the 
    velocity, and a nominal timesteps-per-day.
    FIXME: the timesteps per day were used to calculate the fixed budget lawnmower, they do not belong here."""
    if geo == None:
        geo = {}
    farm = create_wbf({"typename": typename})
    geo["velocity"] = 1
    geo["width"], geo["height"] = farm.width, farm.height
    geo["xmin"], geo["ymin"] = farm.owner_area[0], farm.owner_area[1]
    geo["xmax"], geo["ymax"] = farm.owner_area[2] - 1, farm.owner_area[3] - 1
    geo["timesteps-per-day"] = {"Miniberry-10": 0.4 * 100, "Miniberry-30": 0.4 * 900,
        "Miniberry-100": 0.4 * 10000, "Waterberry": 0.4 * 12000000}[typename]
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
