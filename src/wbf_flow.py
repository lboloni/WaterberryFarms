"""The flow helpers of the Waterberry Farms experiment-flow notebooks. The
generic helpers are implemented by the ExpRunFlow library (exprunflow.flow),
the builders of the flow entries are specific to Waterberry Farms."""

from exp_run_config import Config
from exprunflow.flow import *
from exprunflow.replication import derive_seed, replication_entries


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
        collection_experiment, collection_run, creation_style, config=None, skip_completed=False):
    """Build the environment, simulation, and figure phases for MRMR 2027. For a replicated collection, 
    skip_completed reuses the replications that already have a completed result with the same 
    configuration (exprunflow.replication.is_completed), e.g. to resume an interrupted flow."""
    if config is None:
        config = Config()
    collection = config.get_experiment(
        collection_experiment, collection_run, create_data_dir=False)
    if collection.get("aggregate") is not None:
        if skip_completed:
            # in memory only: setting an item of an Experiment would save it into its (absent) result directory
            values = collection.values if isinstance(getattr(collection, "values", None), dict) else collection
            values["skip-completed"] = True
        return replication_entries(collection, MRMR_APPLIERS, creation_style, config)
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


# Replications: the same runs with varied map and behavior seeds. The mechanism is in the ExpRunFlow 
# library (exprunflow.replication, DESIGN-Replication.md); the project supplies what a seed changes
# (DESIGN-MultiSeedEvaluation.md).

def apply_map_seed(context, map_seed):
    """The map seed: a variant of the run's environment with the seed of its generated disease map, 
    precomputed before the replications"""
    base = context.base
    env_family, env_run = base["exp_environment"], base["run_environment"]
    environment = context.config.get_experiment(env_family, env_run, create_data_dir=False)
    if not environment["tylcv-generated"]:
        raise Exception(f"The map seed can only be varied on a generated map, {env_family}/{env_run} has none")
    name = context.variant(env_family, env_run, {"tylcv-generated-seed": map_seed}, f"{env_run}-m{map_seed}",
                           queue=True)
    return {"run_environment": name}


def apply_behavior_seed(context, behavior_seed):
    """The behavior seed: every robot whose policy has a seed gets one derived from the behavior seed and 
    its position in the team, so robot i of every approach has the same seed (common random numbers)"""
    robots = []
    for index, values in enumerate(context.base["robots"]):
        values = dict(values)
        extra = dict(values.get("exp-policy-extra-parameters") or {})
        policy = context.config.get_experiment(values["exp-policy"], values["run-policy"], create_data_dir=False)
        if "seed" in extra or "seed" in policy:
            extra["seed"] = derive_seed(behavior_seed, index)
            values["exp-policy-extra-parameters"] = extra
        robots.append(values)
    return {"robots": robots}


MRMR_APPLIERS = {"map-seed": apply_map_seed, "behavior-seed": apply_behavior_seed}
