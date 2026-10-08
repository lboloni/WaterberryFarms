"""Small helpers shared by Waterberry Farms experiment-flow notebooks."""

import json
import pathlib
import shutil

import yaml

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


def executed_notebook_path(entry, notebooks_path, notebook_root=REPOSITORY_ROOT):
    """Return where run_notebook retains the executed copy of an entry."""
    stem = (pathlib.Path(notebook_root) / entry["notebook"]).stem
    return (pathlib.Path(notebooks_path)
            / f"{stem}_{entry['experiment']}_{entry['run']}.ipynb")


def run_notebook(
        entry, expruns_path, results_path, notebooks_path,
        notebook_root=REPOSITORY_ROOT, executor=None):
    """Execute one flow entry and retain the executed notebook."""
    if executor is None:
        import papermill
        executor = papermill.execute_notebook

    notebook_path = pathlib.Path(notebook_root) / entry["notebook"]
    output_path = executed_notebook_path(entry, notebooks_path, notebook_root)
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
    """Run an ordered notebook queue with one overall progress bar. Before
    each stage it displays a link to the stage's executed notebook, which
    papermill updates while the stage runs."""
    from html import escape
    from IPython.display import HTML, display

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
            notebook = executed_notebook_path(entry, notebooks_path).resolve()
            display(HTML(
                f"<p>*** {escape(entry['name'])}: "
                f'<a href="{notebook.as_uri()}">{escape(str(notebook))}</a></p>'))
            notebook_runner(
                entry, expruns_path, results_path, notebooks_path)
            progress.update(1)
            progress.set_postfix_str(notebooks_left(
                len(entries) - progress.n))
    finally:
        progress.close()


def read_executed_notebook(path):
    """Return the papermill duration and error of an executed notebook."""
    with pathlib.Path(path).open() as handle:
        notebook = json.load(handle)
    error = None
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("papermill", {}).get("exception"):
            for output in cell.get("outputs", []):
                if output["output_type"] == "error":
                    error = f'{output["ename"]}: {output["evalue"]}'
    return notebook["metadata"]["papermill"].get("duration"), error


def get_flow_report(entries, results_path, notebooks_path, flow_error=None):
    """Return completion information for an executed notebook flow. A stage
    is complete if its exprun.yaml contains time_done. The duration and error
    of a stage come from the papermill metadata of its executed notebook."""
    results_path = pathlib.Path(results_path).resolve()
    stages = []
    for entry in entries:
        directory = results_path / entry["experiment"] / entry["run"]
        provenance = directory / "exprun.yaml"
        complete = False
        if provenance.exists():
            with provenance.open() as handle:
                values = yaml.safe_load(handle)
            complete = Config.TIME_DONE in values
        notebook = executed_notebook_path(entry, notebooks_path)
        duration, error = None, None
        if notebook.exists():
            duration, error = read_executed_notebook(notebook)
        else:
            notebook = None
        stages.append({
            "name": entry["name"],
            "directory": directory,
            "complete": complete,
            "notebook": notebook,
            "duration": duration,
            "error": error,
        })
    completed = [stage for stage in stages if stage["complete"]]
    missing = [stage for stage in stages if not stage["complete"]]
    return {
        "successful": flow_error is None and not missing,
        "all-results-present": not missing,
        "stages": stages,
        "completed": completed,
        "missing": missing,
        "flow-directory": results_path.parent,
        "results-directory": results_path,
        "error": flow_error,
    }


def format_duration(seconds):
    """Format a duration in seconds as e.g. '1 min 48 s'."""
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {seconds} s"
    return f"{seconds} s"


def display_flow_report(
        entries, results_path, notebooks_path, final_results_path,
        flow_error=None, preview_files=()):
    """Display flow status, a per-stage table, artifact links, and selected
    PNG previews."""
    from html import escape
    from IPython.display import HTML, Image, display, display_pdf

    report = get_flow_report(entries, results_path, notebooks_path, flow_error)
    final_results_path = pathlib.Path(final_results_path).resolve()

    def directory_link(label, path):
        path = pathlib.Path(path).resolve()
        return (
            f'<p><strong>{escape(label)}:</strong> '
            f'<a href="{path.as_uri()}">{escape(str(path))}</a></p>')

    if report["successful"]:
        status = "Flow successfully executed."
    elif report["completed"]:
        status = (
            f'Flow has partial results: {len(report["completed"])} of '
            f'{len(entries)} stages completed.')
    else:
        status = "Flow did not produce any completed stage results."

    details = [f"<h2>Flow result</h2><p><strong>{escape(status)}</strong></p>"]
    if flow_error is not None:
        details.append(
            f'<p><strong>Error:</strong> {escape(repr(flow_error))}</p>')
    details.append(directory_link(
        "Flow data directory", report["flow-directory"]))
    details.append(directory_link(
        "All results", report["results-directory"]))
    details.append(directory_link("Final results", final_results_path))

    rows = []
    for stage in report["stages"]:
        if stage["complete"]:
            stage_status = "complete"
        elif stage["error"] is not None:
            stage_status = "failed"
        elif stage["notebook"] is not None:
            stage_status = "incomplete"
        else:
            stage_status = "not run"
        duration = "" if stage["duration"] is None \
            else format_duration(stage["duration"])
        notebook = "" if stage["notebook"] is None else (
            f'<a href="{stage["notebook"].as_uri()}">'
            f'{escape(stage["notebook"].name)}</a>')
        name = escape(stage["name"])
        if stage["error"] is not None:
            name += f'<br><small>{escape(stage["error"])}</small>'
        rows.append(
            f"<tr><td>{name}</td><td>{stage_status}</td>"
            f"<td>{duration}</td><td>{notebook}</td>"
            f'<td><a href="{stage["directory"].as_uri()}">results</a></td></tr>')
    total = sum(stage["duration"] for stage in report["stages"]
                if stage["duration"] is not None)
    details.append(
        "<h3>Stages</h3><table><tr><th>Stage</th><th>Status</th>"
        "<th>Duration</th><th>Executed notebook</th><th>Results</th></tr>"
        f'{"".join(rows)}</table>'
        f"<p><strong>Total stage time:</strong> {format_duration(total)}</p>")
    display(HTML("".join(details)))

    pdfs = sorted(final_results_path.rglob("*.pdf")) \
        if final_results_path.exists() else []
    if pdfs:
        links = "".join(
            f'<li><a href="{path.as_uri()}">'
            f'{escape(str(path.relative_to(final_results_path)))}</a></li>'
            for path in pdfs)
        display(HTML(
            f"<h3>Generated figures</h3><ul>{links}</ul>"))

    for preview in preview_files:
        if isinstance(preview, dict):
            path = pathlib.Path(preview["path"]).resolve()
            description = preview["description"]
        else:
            path = pathlib.Path(preview).resolve()
            description = None
        if path.exists():
            heading = f"<h3>{escape(path.name)}</h3>"
            if description is not None:
                heading += f"<p>{escape(description)}</p>"
            display(HTML(heading))
            if path.suffix.lower() == ".png":
                display(Image(filename=str(path)))
            else:
                display_pdf(path.read_bytes(), raw=True)
    return report
