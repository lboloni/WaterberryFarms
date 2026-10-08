"""
estimator_benchmark.py

A benchmark of estimators: observation sets are generated once (random samples or robot trajectories)
and replayed to every estimator, which is evaluated at checkpoints on accuracy, disease classification,
calibration and cost. See DESIGN-ESTIMATORS.md, Section 9.
"""

import pathlib
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from exp_run_config import Config
from information_model import DISEASED, im_score_weighted_asymmetric
from policy import RandomWaypointPolicy
from robot import Robot
from wbf_helper import create_estimator, create_wbfe, generate_fixed_budget_lawnmower, get_geometry, precompute_environment
from wbf_simulate import simulate_1day


class RecordingEstimator:
    """Records the observations of a simulation, in order"""
    def __init__(self):
        self.observations = []

    def add_observation(self, observation):
        self.observations.append(observation)

    def proceed(self, delta_t):
        pass


class NoScore:
    def score(self, environment, estimator):
        return 0.0


def random_observations(environment, typename, count, seed):
    """count distinct cells of the owner's area, drawn uniformly in a random order"""
    geo = get_geometry(typename)
    cells = [(x, y) for x in range(geo["xmin"], geo["xmax"] + 1) for y in range(geo["ymin"], geo["ymax"] + 1)]
    random = np.random.default_rng(seed)
    observations = []
    for t, index in enumerate(random.choice(len(cells), size=min(count, len(cells)), replace=False)):
        observation = environment.get_observation([cells[index][0], cells[index][1], t])
        observation["robot"] = "random"
        observations.append(observation)
    return observations


def trajectory_observations(environment, typename, policy, timesteps):
    """The observations of a robot following the policy for the timesteps, starting at the corner of the owner's area"""
    geo = get_geometry(typename)
    robot = Robot("robot", geo["xmin"], geo["ymin"], 0)
    robot.assign_policy(policy)
    recorder = RecordingEstimator()
    simulate_1day(environment=environment, robots=[robot], estimator=recorder, evaluator=NoScore(),
                  timesteps=timesteps, estimator_interval=timesteps)
    return recorder.observations


def generate_observations(sampler, environment, typename, seed):
    """An observation set of the sampler: {"kind": "random", "count": n} or
    {"kind": "random-waypoint" | "lawnmower", "timesteps": n}"""
    geo = get_geometry(typename)
    if sampler["kind"] == "random":
        return random_observations(environment, typename, sampler["count"], seed)
    if sampler["kind"] == "random-waypoint":
        policy = RandomWaypointPolicy(geo["velocity"], [geo["xmin"], geo["ymin"]], [geo["xmax"], geo["ymax"]], seed)
        return trajectory_observations(environment, typename, policy, sampler["timesteps"])
    if sampler["kind"] == "lawnmower":
        policy = generate_fixed_budget_lawnmower({"budget": sampler["timesteps"], "policy-name": "lawnmower"}, {"typename": typename})
        return trajectory_observations(environment, typename, policy, sampler["timesteps"])
    raise Exception(f"Unknown sampler {sampler['kind']}")


def sampler_name(sampler):
    return f"{sampler['kind']}-{sampler.get('count', sampler.get('timesteps'))}"


FIELDS = [("tylcv", "my_tomato_mask", True), ("ccr", "my_strawberry_mask", True), ("soil", "my_soil_mask", False)]


def field_metrics(truth_field, im_field, mask, disease):
    """The metrics of the estimate of a field on the cells of the mask"""
    truth, value = truth_field.value[mask], im_field.value[mask]
    error = np.abs(value - truth)
    metrics = {"mae": error.mean(), "rmse": np.sqrt((error**2).mean())}
    confidence = im_field.confidence()[mask]
    metrics["uncertainty-error-correlation"] = (stats.spearmanr(1 - confidence, error).statistic
        if np.ptp(confidence) > 0 and np.ptp(error) > 0 else np.nan)
    if im_field.UNCERTAINTY == "std":
        std = np.maximum(im_field.uncertainty[mask], 1e-9)
        metrics["coverage-1sd"] = np.mean(error <= std)
        metrics["coverage-2sd"] = np.mean(error <= 2 * std)
        metrics["gaussian-nll"] = np.mean(0.5 * np.log(2 * np.pi * std**2) + error**2 / (2 * std**2))
    if not disease:
        return metrics
    # the asymmetric error of the experiments' score, which penalizes the missed disease
    metrics["asymmetric-error"] = -im_score_weighted_asymmetric(truth_field, im_field, 1.0, 10.0, mask)
    diseased = truth < DISEASED
    native = im_field.probability is not None
    probability = im_field.probability[mask] if native else np.clip(2.0 * (1.0 - value), 0, 1)
    predicted = probability >= 0.5
    metrics["native-probability"] = float(native)
    metrics["recall"] = np.sum(predicted & diseased) / diseased.sum() if diseased.any() else np.nan
    metrics["precision"] = np.sum(predicted & diseased) / predicted.sum() if predicted.any() else np.nan
    metrics["f1"] = (2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
                     if metrics["precision"] + metrics["recall"] > 0 else np.nan)
    metrics["auc"] = roc_auc_score(diseased, probability) if 0 < diseased.sum() < len(diseased) else np.nan
    metrics["brier"] = np.mean((probability - diseased) ** 2)
    p_true = np.clip(np.where(diseased, probability, 1 - probability), 1e-6, 1)
    metrics["nll"] = -np.mean(np.log(p_true))
    metrics["confident-errors"] = np.mean(p_true < 0.1)
    bins = np.minimum((probability * 10).astype(int), 9)
    metrics["ece"] = sum(np.sum(bins == b) / len(bins) * abs(probability[bins == b].mean() - diseased[bins == b].mean())
                         for b in range(10) if np.any(bins == b))
    return metrics


def evaluate(estimator, environment, observations, checkpoints, stop_at):
    """Replays the observations to the estimator, and returns the rows of the metrics at every checkpoint
    (number of observations). An estimator whose proceed takes longer than stop_at seconds is stopped."""
    rows = []
    added = 0
    for checkpoint in checkpoints:
        if checkpoint > len(observations):
            break
        for observation in observations[added:checkpoint]:
            estimator.add_observation(observation)
        added = checkpoint
        start = time.perf_counter()
        estimator.proceed(1)
        seconds = time.perf_counter() - start
        rows.append({"field": "all", "observations": checkpoint, "metric": "proceed-seconds", "value": seconds})
        for name, mask_name, disease in FIELDS:
            mask = getattr(environment, mask_name)
            if mask.any():
                for metric, value in field_metrics(getattr(environment, name), getattr(estimator, f"im_{name}"), mask, disease).items():
                    rows.append({"field": name, "observations": checkpoint, "metric": metric, "value": value})
        if seconds > stop_at:
            rows.append({"field": "all", "observations": checkpoint, "metric": "stopped", "value": 1.0})
            break
    return rows


def benchmark_environment(environment_run, time_start):
    """The environment of the scenario, precomputed and advanced to the start time"""
    exp_env = precompute_environment(environment_run)
    _, environment = create_wbfe(exp_env)
    environment.proceed(time_start)
    return exp_env, environment


def run_estimator_benchmark(exp):
    """Runs the estimator benchmark of an estimator-benchmark exp/run; saves and returns the results table"""
    data_dir = pathlib.Path(exp["data_dir"])
    (data_dir / "observations").mkdir(exist_ok=True)
    rows = []
    for environment_run in exp["environment-runs"]:
        exp_env, environment = benchmark_environment(environment_run, exp["time-start-environment"])
        geometry = get_geometry(exp_env["typename"])
        for sampler in exp["samplers"]:
            for repetition in range(exp["repetitions"]):
                path = data_dir / "observations" / f"{environment_run}-{sampler_name(sampler)}-{repetition}.pickle"
                if not path.exists():
                    with open(path, "wb") as f:
                        pickle.dump(generate_observations(sampler, environment, exp_env["typename"], repetition), f)
                with open(path, "rb") as f:
                    observations = pickle.load(f)
                for estimator_run in exp["estimator-runs"]:
                    estimator = create_estimator(Config().get_experiment("estimator", estimator_run, create_data_dir=False), geometry)
                    print(f"{environment_run} {sampler_name(sampler)} {repetition} {estimator_run}")
                    for row in evaluate(estimator, environment, observations, exp["checkpoints"], exp["stop-at"]):
                        rows.append({"environment": environment_run, "sampler": sampler_name(sampler),
                                     "repetition": repetition, "estimator": estimator_run} | row)
    results = pd.DataFrame(rows)
    results.to_pickle(data_dir / "results.pickle")
    results.to_csv(data_dir / "results.csv", index=False)
    return results
