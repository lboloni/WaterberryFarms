"""Small helpers shared by Waterberry Farms experiment-flow notebooks."""

import pathlib
import shutil

from exp_run_config import Config


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent


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


def setup_flow(flow_name, experiment_families, flows_path=None, config=None):
    """Create an external flow workspace and point Config at it."""
    if config is None:
        config = Config()
    if flows_path is None:
        flows_path = config["flows_path"]

    source_path = pathlib.Path(config.get_exprun_path())
    flow_path = pathlib.Path(flows_path).expanduser() / flow_name
    expruns_path = flow_path / "expruns"
    results_path = flow_path / "results"
    notebooks_path = flow_path / "executed-notebooks"
    expruns_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    notebooks_path.mkdir(parents=True, exist_ok=True)

    for family in experiment_families:
        shutil.copytree(
            source_path / family,
            expruns_path / family,
            dirs_exist_ok=True,
        )

    config.set_exprun_path(expruns_path)
    config.set_results_path(results_path)
    return expruns_path, results_path, notebooks_path


def run_notebook(
        entry, expruns_path, results_path, notebooks_path,
        notebook_root=REPOSITORY_ROOT, executor=None):
    """Execute one flow entry and retain the executed notebook."""
    if executor is None:
        import papermill
        executor = papermill.execute_notebook

    notebook_path = pathlib.Path(notebook_root) / entry["notebook"]
    output_name = (
        f"{notebook_path.stem}_{entry['experiment']}_{entry['run']}.ipynb")
    output_path = pathlib.Path(notebooks_path) / output_name
    parameters = {
        "experiment": entry["experiment"],
        "run": entry["run"],
        "creation_style": entry["creation_style"],
        "expruns_path": pathlib.Path(expruns_path).as_posix(),
        "results_path": pathlib.Path(results_path).as_posix(),
    }
    executor(
        notebook_path,
        output_path,
        cwd=notebook_path.parent,
        parameters=parameters,
    )
    return output_path


def run_flow(
        entries, expruns_path, results_path, notebooks_path,
        notebook_runner=run_notebook, progress_factory=None):
    """Run an ordered notebook queue with one overall progress bar."""
    def notebooks_left(count):
        unit = "notebook" if count == 1 else "notebooks"
        return f"{count} {unit} left"

    if progress_factory is None:
        from tqdm import tqdm
        progress_factory = tqdm

    progress = progress_factory(
        total=len(entries), desc="Overall flow", unit="notebook")
    progress.set_postfix_str(notebooks_left(len(entries)))
    try:
        for entry in entries:
            remaining = len(entries) - progress.n
            progress.set_postfix_str(
                f"{notebooks_left(remaining)}; current: {entry['name']}")
            print(f"*** {entry['name']}", flush=True)
            notebook_runner(
                entry, expruns_path, results_path, notebooks_path)
            progress.update(1)
            progress.set_postfix_str(notebooks_left(
                len(entries) - progress.n))
    finally:
        progress.close()
