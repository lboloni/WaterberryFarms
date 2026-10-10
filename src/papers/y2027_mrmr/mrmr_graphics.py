"""
mrmr_graphics.py

Helper functions for creating the graphics for the MRMR paper
"""

from exp_run_config import Config
Config.PROJECTNAME = "WaterBerryFarms"

import pathlib
import matplotlib
import matplotlib.pyplot as plt
import wbf_figures
import logging
import numpy as np
import gzip as compress
import pickle
import pprint
import tqdm


from water_berry_farm import voi_credits

logging.getLogger("fontTools").setLevel(logging.WARNING)


def figure_paths(output_filename):
    """Return the publication PDF and notebook-preview PNG paths."""
    pdf_path = pathlib.Path(output_filename)
    return pdf_path, pdf_path.with_suffix(".png")


def save_figure(fig, output_filename):
    """Save a figure as a publication PDF and an inline-preview PNG."""
    pdf_path, png_path = figure_paths(output_filename)
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    print(f"Done saving to {pdf_path} and {png_path}")


def build_figure_previews(figure_experiment, figure_runs, config=None):
    """Describe every generated PNG for the MRMR flow report."""
    if config is None:
        config = Config()
    previews = []
    for run in figure_runs:
        exp = config.get_experiment(
            figure_experiment, run, create_data_dir=False)
        sources = [
            f'{exp["source-experiment"]}/{source}'
            for source in exp["source-runs"]]
        if sources:
            origin = f'Source: {", ".join(sources)}.'
        else:
            origin = f'Source: geometry declared by {figure_experiment}/{run}.'
        for path in sorted(pathlib.Path(exp["data_dir"]).glob("*.png")):
            description = exp["name"]
            if run.startswith("detection-map-"):
                description = (
                    "Robot trajectories and detection locations for "
                    f'{exp["source-runs"][0]}')
            elif run.startswith("agent-voi-"):
                description = (
                    "Per-agent and total VoI bar graph for "
                    f'{exp["source-runs"][0]}')
            elif run.startswith("communication-cost"):
                description = (
                    "Communication cost (cumulative and per message type) of "
                    f'{", ".join(exp["source-runs"])}')
            elif run.startswith("comparison-"):
                description = (
                    "VoI comparison bar graphs for the "
                    f'{run.removeprefix("comparison-")} environment')
            elif "output-filename-prefix" in exp:
                suffix = path.stem.removeprefix(
                    f'{exp["output-filename-prefix"]}_')
                robot, time = suffix.rsplit("_", 1)
                description = (
                    f'Replanning snapshot for robot {robot} at t={time}')
            previews.append({
                "path": path,
                "description": (
                    f'{description}. {origin} '
                    f'Figure: {figure_experiment}/{run}.'),
            })
    return previews


def load_back_results(experiment, listruns):
    """Loads back all the results of the experiment runs specified into a list"""
    all_results = {}

    for run in tqdm.tqdm(listruns):
        exp = Config().get_experiment(experiment, run)
        # pprint.pprint(exp)

        resultsfile = pathlib.Path(exp["data_dir"], "results.pickle")
        if not resultsfile.exists():
            print(f"Results file does not exist:\n{resultsfile}")
            print("Run the notebook Run-1Robot1Day with the same exp/run to create it.")
            raise Exception("Nothing to do.")

        # load the results file
        with compress.open(resultsfile, "rb") as f:
            results = pickle.load(f)    
        all_results[run] = results
    return all_results

def show_robot_with_plan(
        expall, scenario, results, robotno, t, output_filename=None):
    """Visualize the plan of the robot at a certain time point"""

    ROBOT_COLORS = ["#E69F00", "#56B4E9", "#009E73"]
    robot_color = ROBOT_COLORS[2]

    robot = results["robots"][robotno]
    observations = [o[robotno] for o in results["observations"]]
    observations = observations[0:int(t)]
    if observations:
        print(f"Last observations: {observations[-1]}")

    oldplan = robot.oldplans[t]


    if output_filename is None:
        output_filename = f"plans_{scenario}_{robot.name}_{t}.pdf"

    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    wbf_figures.show_env_tylcv(results, ax)
    # obs = observations[1:int(t)]
    # color = "blue"
    wbf_figures.show_individual_robot_path(results, ax, robot=robot, observations=observations, pathcolor=robot_color, pathwidth=1,  draw_robot=False, robotcolor=robot_color, from_obs=0, to_obs=int(t))

    # add the plan
    # print(f"Oldplan beginning: {oldplan[0]}")
    planx = [a["x"] for a in oldplan]
    plany = [a["y"] for a in oldplan]
    ax.add_line(matplotlib.lines.Line2D(planx, plany, color = robot_color, linestyle=":", linewidth=1))

    # visualize the position of the robot
    #ax.add_patch(matplotlib.patches.Circle((observations[int(t)]["x"], observations[int(t)]["y"]), radius=3, facecolor=robot_color))
    if observations:
        ax.add_patch(matplotlib.patches.Circle((observations[-1]["x"], observations[-1]["y"]), radius=3, facecolor=robot_color))
    # ax.add_patch(matplotlib.patches.Circle((oldplan[0]["x"], oldplan[0]["y"]), radius=3, facecolor="yellow"))

    ax.set_title(f"{robot.name} at t={int(t)}")
    filepath = pathlib.Path(expall.data_dir(), output_filename)
    save_figure(fig, filepath)
    plt.close(fig)

def show_robot_trajectories_and_detections(
        exp_dest, name, results, robot_colors, lookup,
        output_filename=None):
    """ Visualize detection paths for all running scenarios. Create a graph for the visualization of the paths, with the visualization of the detections
    exp_dest: the exprun whose data dir the figures are going to be put
    """
    if output_filename is None:
        output_filename = f"detections-map-{name}.pdf"
    fig_file = pathlib.Path(exp_dest.data_dir(), output_filename)
    figure_files = figure_paths(fig_file)
    if all(path.exists() for path in figure_files):
        print(f"{fig_file} and its PNG preview exist, skipping.")
        return
    fig, ax = plt.subplots(1,1, figsize=(3, 3))
    wbf_figures.show_env_tylcv(results, ax)
    if lookup and name in lookup:
        ax.set_title(lookup[name])
    else:
        print(f"Missing name map:\n{name}")
        ax.set_title(name)
    custom_lines = []
    labels = []

    for i, robot in enumerate(results["robots"]):
        color = robot_colors[i % len(results["robots"])]
        observations = [o[i] for o in results["observations"]]
        wbf_figures.show_individual_robot_path(results, ax, robot=robot, observations=observations, pathcolor=color, draw_robot=False)
        wbf_figures.show_individual_robot_detections(results, ax, robotno=i, detection_color=color, radius=0.5)
        # adding to the legend
        custom_lines.append(matplotlib.lines.Line2D([0], [0], color=color, lw=2))
        labels.append(robot.name)

    # Add both automatic and manual entries to the legend
    ax.legend(handles=[*custom_lines], labels=labels, ncol=3, bbox_to_anchor=(0.5, -0.1), loc="upper center", fontsize="9", columnspacing=0.5, labelspacing=0.5)
    #bbox_to_anchor=(0.5, 0))
             #, loc="upper center")

    # fig.legend(handles, labels, ncol=len(exps)+1,
    #        bbox_to_anchor=(0.5, 0), loc="upper center")

    save_figure(fig, fig_file)
    plt.close(fig)

VOI_LABELS = {"voi-absolute": "Absolute VoI", "voi-expected": "Expected VoI",
              "voi-ignorance": "VoI (cost of ignorance)", "voi-estimator": "Estimator-based VoI",
              "voi": "Estimator-based VoI"}


def agentwise_voi(results, variant = "voi-absolute"):
    """The VoI credited to each robot over the run, in robot order, followed by the estimator update 
    terms under "Estimator" if they are not zero (DESIGN-VOI.md, Section 4)"""
    credits = voi_credits(results, variant)
    values = {name: float(credits[name].sum()) for name in results["robot-names"]}
    estimator = float(credits[None].sum())
    if estimator != 0.0:
        values["Estimator"] = estimator
    return values


def show_agentwise_voi(
        exp_dest, name, results, robot_colors, variant = "voi-absolute", output_filename=None):
    """Create a bargraph with the VoI credited to each agent and their total in the specific run
    exp_dest: the exprun whose data dir the figures are going to be put
    """   
    if output_filename is None:
        output_filename = f"voi-bar-{name}.pdf"
    fig_file = pathlib.Path(exp_dest.data_dir(), output_filename)
    figure_files = figure_paths(fig_file)
    if all(path.exists() for path in figure_files):
        print(f"{fig_file} and its PNG preview exist, skipping.")
        return
    fig, ax = plt.subplots(1,1, figsize=(3, 1.6))
    values = agentwise_voi(results, variant)
    for i, (label, value) in enumerate(values.items()):
        color = "silver" if label == "Estimator" else robot_colors[i % len(robot_colors)]
        ax.bar(label, value, color=color)
    ax.bar("Total", sum(values.values()), color="gray")
    ax.set_ylabel(VOI_LABELS.get(variant, variant))
    # robot names such as robot-1 do not fit side by side
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    save_figure(fig, fig_file)
    plt.close(fig)


def show_comparative_voi(
        exp_dest, filename, all_results, lookup, name_colors,
        variants = ("voi-absolute", "voi-estimator"), output_filename=None):
    """Create a comparative bargraph of the final VoI of the runs in all_results, one panel per variant"""
    fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3), squeeze=False)
    for ax, variant in zip(axes[0], variants):
        for i, policyname in enumerate(all_results):
            if policyname in lookup:
                name = lookup[policyname]
            else: 
                print(f"No short name for:\n{policyname}")
                name = policyname
            ax.bar(name, float(all_results[policyname]["score"][variant]), 
                   color=name_colors[i % len(name_colors)])
        ax.set_ylabel(VOI_LABELS.get(variant, variant))
        ax.axhline(0, color="black", linewidth=0.5)
        # rotate the labels, as they don't fit
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
    fig.tight_layout()
    if output_filename is None:
        output_filename = f"comparative-voi-{filename}.pdf"
    save_figure(
        fig, pathlib.Path(exp_dest.data_dir(), output_filename))
    plt.close(fig)


MESSAGE_TYPES = ["location", "ep-offer", "ep-bid", "ep-award", "ep-completed"]


def show_communication_cost(
        exp_dest, filename, all_results, lookup, name_colors, output_filename=None):
    """Create a figure of the communication cost of the runs in all_results (DESIGN-COMMUNICATION.md, 
    Section 2.1): left, the cumulative transmitted kB over time; right, the transmitted kB per 
    message type, on a logarithmic scale as the location messages dominate"""
    fig, (ax_time, ax_type) = plt.subplots(1, 2, figsize=(7, 3))
    summaries = {name: results["communication-summary"] for name, results in all_results.items()}
    types = [t for t in MESSAGE_TYPES if any(t in s["per-type"] for s in summaries.values())]
    types += sorted({t for s in summaries.values() for t in s["per-type"]} - set(types), key=str)
    width = 0.8 / max(1, len(summaries))
    for i, (name, summary) in enumerate(summaries.items()):
        label = lookup.get(name, name)
        color = name_colors[i % len(name_colors)]
        cumulative = np.cumsum(summary["bytes-per-timestep"]) / 1000
        ax_time.plot(np.arange(len(cumulative)), cumulative, color=color, label=label)
        values = [summary["per-type"].get(t, {}).get("bytes-transmitted", 0) / 1000 for t in types]
        ax_type.bar(np.arange(len(types)) + (i - (len(summaries) - 1) / 2) * width, values, width, 
                    color=color, label=label)
    ax_time.set_xlabel("Timestep")
    ax_time.set_ylabel("Transmitted (kB)")
    ax_time.legend(fontsize=8)
    ax_type.set_xticks(np.arange(len(types)))
    ax_type.set_xticklabels([str(t) for t in types], rotation=30, ha="right", fontsize=8)
    ax_type.set_yscale("log")
    ax_type.set_ylabel("Transmitted (kB)")
    fig.tight_layout()
    if output_filename is None:
        output_filename = f"communication-cost-{filename}.pdf"
    save_figure(fig, pathlib.Path(exp_dest.data_dir(), output_filename))
    plt.close(fig)


# Figures of the replicated comparisons that are specific to MRMR, drawn from the tidy tables of the 
# aggregation (exprunflow.aggregate). The generic ones (bars, ranges, series, paired differences) are in 
# exprunflow.plots. Both take a `where` filter, e.g. {"map-size": 200}.

METRIC_LABELS = {"voi-absolute": "Absolute VoI", "voi-estimator": "Estimator-based VoI",
                 "diseased-found": "Diseased plants found", "cells-observed": "Cells observed",
                 "bytes-transmitted": "Transmitted (bytes)", "bytes-transmitted-cumulative": "Transmitted (kB)"}


def show_replicated_communication(tables, scenarios, approach, lookup, colors, output_path, where=None):
    """The communication cost of an approach: left, the cumulative transmitted kB over time, mean with 
    95% CI band, per scenario; right, the transmitted kB per message type, mean with 95% CI (log scale)"""
    from exprunflow.aggregate import confidence_interval
    from exprunflow.plots import save_figure, select
    from .mrmr_metrics import MESSAGE_TYPES
    series = select(tables["summary-series"], where, approach=approach, metric="bytes-transmitted-cumulative")
    summary = select(tables["summary"], where, approach=approach)
    fig, (ax_time, ax_type) = plt.subplots(1, 2, figsize=(7, 3))
    width = 0.8 / len(scenarios)
    for i, scenario in enumerate(scenarios):
        color = colors[i % len(colors)]
        frame = select(series, scenario=scenario).sort_values("timestep")
        ci = np.array([confidence_interval(row) for _, row in frame.iterrows()], dtype=float) / 1000
        ax_time.fill_between(frame["timestep"], ci[:, 0], ci[:, 1], color=color, alpha=0.35, linewidth=0)
        ax_time.plot(frame["timestep"], frame["mean"] / 1000, color=color, label=lookup.get(scenario, scenario))
        for j, message_type in enumerate(MESSAGE_TYPES):
            row = select(summary, scenario=scenario, metric=f"bytes-{message_type}").iloc[0]
            low, high = confidence_interval(row)
            err = [[(row["mean"] - low) / 1000], [(high - row["mean"]) / 1000]] if not np.isnan(low) else None
            ax_type.bar(j + (i - (len(scenarios) - 1) / 2) * width, row["mean"] / 1000, width, color=color,
                        yerr=err, capsize=2)
    ax_time.set_xlabel("Timestep")
    ax_time.set_ylabel("Transmitted (kB)")
    ax_time.legend(fontsize=8)
    ax_type.set_xticks(range(len(MESSAGE_TYPES)))
    ax_type.set_xticklabels(MESSAGE_TYPES, rotation=30, ha="right", fontsize=8)
    ax_type.set_yscale("log")
    ax_type.set_ylabel("Transmitted (kB)")
    fig.tight_layout()
    return save_figure(fig, output_path)


def show_replicated_per_role(tables, scenarios, metric, lookup, colors, output_path, where=None):
    """The metric per robot, by role, one panel per scenario sharing the y axis: the mean over the robots of 
    a role and the replications, with the 95% CI over the replications (of the per-replication role means)"""
    from exprunflow.aggregate import describe
    from exprunflow.plots import save_figure, select
    entities = select(tables["entities"], where, metric=metric)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.5 * len(scenarios), 3), sharey=True, squeeze=False)
    for ax, scenario in zip(axes[0], scenarios):
        frame = select(entities, scenario=scenario)
        per_replication = frame.groupby(["approach", "group", "run"])["value"].mean().reset_index()
        groups = list(per_replication.groupby(["approach", "group"], sort=False))
        for i, ((approach, role), values) in enumerate(groups):
            d = describe(values["value"])
            err = [[d["mean"] - d["ci95_low"]], [d["ci95_high"] - d["mean"]]] if not np.isnan(d["ci95_low"]) else None
            ax.bar(i, d["mean"], color=colors[i % len(colors)], yerr=err, capsize=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{lookup.get(a, a)} {r}" for (a, r), _ in groups], rotation=45, ha="right", fontsize=8)
        ax.set_title(lookup.get(scenario, scenario), fontsize=10)
        ax.set_ylabel(f"{METRIC_LABELS.get(metric, metric)} per robot")
    fig.tight_layout()
    return save_figure(fig, output_path)
