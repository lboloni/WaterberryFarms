"""The flow helpers of the Waterberry Farms experiment-flow notebooks. The
generic helpers are implemented by the ExpRunFlow library (exprunflow.flow),
the builders of the flow entries are specific to Waterberry Farms."""

from exp_run_config import Config
from exprunflow.flow import *
from exprunflow.replication import derive_seed, replication_entries, without_completed


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


def build_marex_2027_flow_entries(
        collection_experiment, collection_run, creation_style, config=None, skip_completed=False,
        replications=None):
    """Build the environment, simulation, and figure phases for MAREX 2027. A collection is single-run 
    (runs and figures drawn from them), replicated (aggregate and figures drawn from it), or both: then the 
    single-run phases come first, followed by the replicated ones (the figures of the aggregate are its 
    aggregate-figures). For a replicated collection, skip_completed reuses the replications that already 
    have a completed result with the same configuration (exprunflow.replication.is_completed), e.g. to 
    resume an interrupted flow; for a single-run collection, it also reuses the completed environments and 
    runs. replications: N runs only the first N replications of every base run (a partial experiment, 
    exprunflow DESIGN-Replication.md, Section 2.6); with skip_completed, a larger N continues it."""
    if config is None:
        config = Config()
    collection = config.get_experiment(
        collection_experiment, collection_run, create_data_dir=False)
    if collection.get("aggregate") is not None and collection.get("runs") is not None:
        values = collection.values if isinstance(getattr(collection, "values", None), dict) else collection
        single = {key: value for key, value in values.items() if key not in ("aggregate", "aggregate-figures")}
        replicated = {key: value for key, value in values.items() if key not in ("runs", "figures")}
        replicated["figures"] = values.get("aggregate-figures", [])
        if skip_completed:
            replicated["skip-completed"] = True
        if replications is not None:
            replicated["replications"] = replications
        entries = (single_run_entries(single, creation_style, config, skip_completed)
                   + replication_entries(replicated, MAREX_APPLIERS, creation_style, config))
        return merge_phases(entries)
    if collection.get("aggregate") is not None:
        # in memory only: setting an item of an Experiment would save it into its (absent) result directory
        values = collection.values if isinstance(getattr(collection, "values", None), dict) else collection
        if skip_completed:
            values["skip-completed"] = True
        if replications is not None:
            values["replications"] = replications
        return replication_entries(collection, MAREX_APPLIERS, creation_style, config)
    return single_run_entries(collection, creation_style, config, skip_completed)


def single_run_entries(collection, creation_style, config, skip_completed=False):
    """The phases of a single-run collection: the environments, the runs, and the figures drawn from them. 
    With skip_completed, the environments and runs that already have a completed result with the same 
    configuration are left out; the figures always run."""
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
            "phase": "prepare",
        })

    for exp, run in zip(runs, run_names):
        entries.append({
            "name": f"Run {run_experiment}/{run}",
            "notebook": exp["input-to-notebook"][0],
            "experiment": run_experiment,
            "run": run,
            "creation_style": creation_style,
            "phase": "run",
        })

    if skip_completed:
        entries = without_completed(entries, config)

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
        # a figure starts from an empty data directory, so that it never keeps files of earlier runs (e.g. the
        # replanning snapshots of a run with another budget)
        entries.append({
            "name": f"Figure {figure_experiment}/{figure_name}",
            "notebook": figure["input-to-notebook"][0],
            "experiment": figure_experiment,
            "run": figure_name,
            "creation_style": "discard-old",
            "phase": "figure",
        })
    return entries


# the order of the phases of a parallel flow (ExpRunFlow docs/DESIGN-Parallelization.md)
PHASE_ORDER = ["prepare", "run", "aggregate", "figure"]


def merge_phases(entries):
    """The entries of a combined collection ordered by phase, stably: the environments of the example runs 
    and of the replications are prepared together, the example runs run together with the replications, 
    and all the figures come after the aggregation. In a parallel flow each phase is then one group."""
    return sorted(entries, key=lambda entry: PHASE_ORDER.index(entry["phase"]))


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


MAREX_APPLIERS = {"map-seed": apply_map_seed, "behavior-seed": apply_behavior_seed}
