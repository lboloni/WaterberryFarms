"""The flow helpers of the Waterberry Farms experiment-flow notebooks. The
generic helpers are implemented by the ExpRunFlow library (exprunflow.flow),
the builders of the flow entries are specific to Waterberry Farms."""

import pathlib

import numpy as np
import yaml

from exp_run_config import Config
from exprunflow.flow import *


def build_flow_entries(
        collection_experiment, collection_run, creation_style, config=None):
    """Build environment, run, visualization, and comparison phases."""
    if config is None:
        config = Config()
    collection = config.get_experiment(
        collection_experiment, collection_run, create_data_dir=False)
    members = [
        config.get_experiment(
            collection_experiment, run, create_data_dir=False)
        for run in collection["tocompare"]
    ]

    entries = []
    environments = []
    for member in members:
        reference = (member["exp_environment"], member["run_environment"])
        if reference not in environments:
            environments.append(reference)
    for experiment, run in environments:
        environment = config.get_experiment(
            experiment, run, create_data_dir=False)
        entries.append({
            "name": f"Precompute {experiment}/{run}",
            "notebook": environment["input-to-notebook"][0],
            "experiment": experiment,
            "run": run,
            "creation_style": creation_style,
        })
    for member, run in zip(members, collection["tocompare"]):
        entries.append({
            "name": f"Run {collection_experiment}/{run}",
            "notebook": member["input-to-notebook"][0],
            "experiment": collection_experiment,
            "run": run,
            "creation_style": creation_style,
        })
    for member, run in zip(members, collection["tocompare"]):
        entries.append({
            "name": f"Visualize {collection_experiment}/{run}",
            "notebook": member["input-to-notebook"][1],
            "experiment": collection_experiment,
            "run": run,
            "creation_style": "exist-ok",
        })
    entries.append({
        "name": f"Compare {collection_experiment}/{collection_run}",
        "notebook": collection["input-to-notebook"][0],
        "experiment": collection_experiment,
        "run": collection_run,
        "creation_style": creation_style,
    })
    return entries


def build_mrmr_2027_flow_entries(
        collection_experiment, collection_run, creation_style, config=None):
    """Build the environment, simulation, and figure phases for MRMR 2027."""
    if config is None:
        config = Config()
    collection = config.get_experiment(
        collection_experiment, collection_run, create_data_dir=False)
    if collection.get("aggregate") is not None:
        return build_mrmr_2027_replicated_entries(collection, creation_style, config)
    run_experiment = collection["run-experiment"]
    figure_experiment = collection["figure-experiment"]
    run_names = collection["runs"]
    runs = [
        config.get_experiment(
            run_experiment, run, create_data_dir=False)
        for run in run_names
    ]

    entries = []
    environments = []
    for run in runs:
        reference = (run["exp_environment"], run["run_environment"])
        if reference not in environments:
            environments.append(reference)
    for experiment, run in environments:
        environment = config.get_experiment(
            experiment, run, create_data_dir=False)
        entries.append({
            "name": f"Precompute {experiment}/{run}",
            "notebook": environment["input-to-notebook"][0],
            "experiment": experiment,
            "run": run,
            "creation_style": creation_style,
        })

    for exp, run in zip(runs, run_names):
        entries.append({
            "name": f"Run {run_experiment}/{run}",
            "notebook": exp["input-to-notebook"][0],
            "experiment": run_experiment,
            "run": run,
            "creation_style": creation_style,
        })

    for figure_name in collection["figures"]:
        figure = config.get_experiment(
            figure_experiment, figure_name, create_data_dir=False)
        if figure["source-experiment"] != run_experiment:
            raise Exception(
                f"Figure {figure_name} refers to experiment "
                f"{figure['source-experiment']}, expected {run_experiment}")
        for source_run in figure["source-runs"]:
            if source_run not in run_names:
                raise Exception(
                    f"Figure {figure_name} refers to undeclared run "
                    f"{source_run}")
        entries.append({
            "name": f"Figure {figure_experiment}/{figure_name}",
            "notebook": figure["input-to-notebook"][0],
            "experiment": figure_experiment,
            "run": figure_name,
            "creation_style": creation_style,
        })
    return entries


# Replications: the same runs with varied map and behavior seeds (DESIGN-MultiSeedEvaluation.md)

def robot_seed(behavior_seed, index):
    """The seed of the robot at the index of the robots list in the replication with the behavior seed. 
    Robot i of every approach gets the same seed (common random numbers)."""
    return int(np.random.SeedSequence([behavior_seed, index]).generate_state(1)[0])


def environment_variant_name(environment_run, map_seed):
    return f"{environment_run}-m{map_seed}"


def replication_name(base_run, map_seed, behavior_seed):
    return f"{base_run}-m{map_seed}-b{behavior_seed}"


def replication_names(aggregate):
    """The replication runs of an aggregate exp/run: every base run with every map and behavior seed"""
    return [replication_name(run, m, b) for run in aggregate["runs"]
            for m in aggregate["map-seeds"] for b in aggregate["behavior-seeds"]]


def write_exprun_variant(family, run, changes, new_run, config=None):
    """Write the run of the family, with the changes to its top-level values, as the new run into the 
    active exp/run path (in a flow, the flow workspace). The variant is written from the run's own file, 
    so it keeps inheriting the defaults of the family."""
    if config is None:
        config = Config()
    family_path = pathlib.Path(config.get_exprun_path(), family)
    with open(family_path / f"{run}.yaml") as f:
        values = yaml.safe_load(f) or {}
    values.update(changes)
    with open(family_path / f"{new_run}.yaml", "w") as f:
        f.write(f"# generated from {family}/{run} by write_exprun_variant, do not edit\n")
        yaml.safe_dump(values, f, sort_keys=False)
    return new_run


def replicated_robots(base, behavior_seed, config):
    """The robots of a base run, with the seed of every robot whose policy has one replaced by its 
    replication seed"""
    robots = []
    for index, values in enumerate(base["robots"]):
        values = dict(values)
        extra = dict(values.get("exp-policy-extra-parameters") or {})
        policy = config.get_experiment(values["exp-policy"], values["run-policy"], create_data_dir=False)
        if "seed" in extra or "seed" in policy:
            extra["seed"] = robot_seed(behavior_seed, index)
            values["exp-policy-extra-parameters"] = extra
        robots.append(values)
    return robots


def build_mrmr_2027_replicated_entries(collection, creation_style, config):
    """Generate the replications declared by the aggregate exp/run of the collection into the active 
    exp/run path, and build their phases: the environment variants, the replications, the aggregation, 
    and the figures"""
    run_experiment = collection["run-experiment"]
    aggregate_experiment = collection["aggregate-experiment"]
    aggregate = config.get_experiment(aggregate_experiment, collection["aggregate"], create_data_dir=False)
    if aggregate["source-experiment"] != run_experiment:
        raise Exception(f"The aggregate {collection['aggregate']} refers to {aggregate['source-experiment']}, "
                        f"expected {run_experiment}")
    entries = []
    environments = {}  # (environment family, run, map seed) -> variant name
    runs = []
    for base_name in aggregate["runs"]:
        base = config.get_experiment(run_experiment, base_name, create_data_dir=False)
        env_family, env_run = base["exp_environment"], base["run_environment"]
        environment = config.get_experiment(env_family, env_run, create_data_dir=False)
        if not environment["tylcv-generated"]:
            raise Exception(f"{base_name}: the map seed can only be varied on a generated map, "
                            f"{env_family}/{env_run} has none")
        for map_seed in aggregate["map-seeds"]:
            key = (env_family, env_run, map_seed)
            if key not in environments:
                environments[key] = write_exprun_variant(env_family, env_run, {"tylcv-generated-seed": map_seed},
                    environment_variant_name(env_run, map_seed), config)
                entries.append({
                    "name": f"Precompute {env_family}/{environments[key]}",
                    "notebook": environment["input-to-notebook"][0],
                    "experiment": env_family,
                    "run": environments[key],
                    "creation_style": creation_style,
                })
            for behavior_seed in aggregate["behavior-seeds"]:
                runs.append(write_exprun_variant(run_experiment, base_name, {
                    "run_environment": environments[key],
                    "robots": replicated_robots(base, behavior_seed, config),
                    "base-run": base_name,
                    "behavior-seed": behavior_seed,
                }, replication_name(base_name, map_seed, behavior_seed), config))
    for run in runs:
        exp = config.get_experiment(run_experiment, run, create_data_dir=False)
        entries.append({
            "name": f"Run {run_experiment}/{run}",
            "notebook": exp["input-to-notebook"][0],
            "experiment": run_experiment,
            "run": run,
            "creation_style": creation_style,
        })
    entries.append({
        "name": f"Aggregate {aggregate_experiment}/{collection['aggregate']}",
        "notebook": aggregate["input-to-notebook"][0],
        "experiment": aggregate_experiment,
        "run": collection["aggregate"],
        "creation_style": creation_style,
    })
    figure_experiment = collection["figure-experiment"]
    for figure_name in collection["figures"]:
        figure = config.get_experiment(figure_experiment, figure_name, create_data_dir=False)
        if figure["source-experiment"] != aggregate_experiment or figure["source-runs"] != [collection["aggregate"]]:
            raise Exception(f"Figure {figure_name} does not draw the aggregate {collection['aggregate']}")
        entries.append({
            "name": f"Figure {figure_experiment}/{figure_name}",
            "notebook": figure["input-to-notebook"][0],
            "experiment": figure_experiment,
            "run": figure_name,
            "creation_style": creation_style,
        })
    return entries
