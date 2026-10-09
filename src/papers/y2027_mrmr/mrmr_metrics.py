"""
mrmr_metrics.py

The metrics of one MRMR run, as plain data (metrics.json), from which the replications are aggregated
without loading their results pickles. See DESIGN-MultiSeedEvaluation.md, Section 5.
"""

import json
import pathlib

import numpy as np

from policy import RandomWaypointPolicy
from water_berry_farm import voi_credits

from .mrmr_policies import MRMR_Contractor, MRMR_Pioneer

METRICS_FILE = "metrics.json"
# the VoI variants recorded, if the run was scored with the VoI score
VOI_VARIANTS = ["voi-absolute", "voi-estimator"]
# the message types whose transmitted bytes are recorded as bytes-<type>
MESSAGE_TYPES = ["location", "ep-offer", "ep-bid", "ep-award", "ep-completed"]


def robot_role(robot):
    """The role of a robot, which is comparable across replications (robot names are not)"""
    if isinstance(robot.policy, MRMR_Pioneer):
        return "pioneer"
    if isinstance(robot.policy, MRMR_Contractor):
        return "contractor"
    if isinstance(robot.policy, RandomWaypointPolicy):
        return "random-waypoint"
    return "lawnmower"


def collect_metrics(exp, results, identification, computation_seconds=None):
    """The metrics of a run: its identification, final scalars, per-robot values, and series sampled at
    the scoring events. identification: run, base-run, scenario, approach, map-seed, behavior-seed"""
    environment = results["wbfe"]
    record = results["estimator"].record
    diseased = (environment.tylcv.value < 1.0) & environment.my_tomato_mask
    cells = list(record.cells.items())
    found = [(cell, rec) for cell, rec in cells if diseased[cell]]
    summary = results["communication-summary"]
    per_type = summary["per-type"]
    events = results["score-events"]
    scored_voi = isinstance(results["score"], dict) and "voi-absolute" in results["score"]

    scalars = {
        "diseased-cells": int(diseased.sum()),
        "diseased-found": len(found),
        "cells-observed": len(cells),
        "messages": summary["messages"],
        "bytes-transmitted": summary["bytes-transmitted"],
        "bytes-delivered": summary["bytes-delivered"],
        "eps-offered": per_type.get("ep-offer", {}).get("messages", 0),
        "eps-awarded": per_type.get("ep-award", {}).get("messages", 0),
        "eps-completed": per_type.get("ep-completed", {}).get("messages", 0),
    }
    for message_type in MESSAGE_TYPES:
        scalars[f"bytes-{message_type}"] = per_type.get(message_type, {}).get("bytes-transmitted", 0)
    if computation_seconds is not None:
        scalars["computation-seconds"] = round(computation_seconds, 3)

    timesteps = [event["timestep"] for event in events]
    first_times = np.array(sorted(rec["first-time"] for _, rec in found))
    cumulative_bytes = np.cumsum(summary["bytes-per-timestep"])
    series = {
        "timestep": timesteps,
        "diseased-found": [int(np.searchsorted(first_times, t, side="right")) for t in timesteps],
        "bytes-transmitted-cumulative": [int(cumulative_bytes[t]) for t in timesteps],
    }

    per_robot = {}
    credits = voi_credits(results, "voi-absolute") if scored_voi else None
    for robot in results["robots"]:
        name = robot.name
        per_robot[name] = {
            "role": robot_role(robot),
            "cells-observed": sum(1 for _, rec in cells if rec["first-robot"] == name),
            "diseased-found": sum(1 for _, rec in found if rec["first-robot"] == name),
            "bytes-transmitted": summary["per-sender"].get(name, {}).get("bytes-transmitted", 0),
        }
        if credits is not None:
            per_robot[name]["voi-absolute"] = float(credits[name].sum())

    if scored_voi:
        for variant in VOI_VARIANTS:
            scalars[variant] = float(results["score"][variant])
            series[variant] = [float(event["score"][variant]) for event in events]

    return {**identification, "robots": [robot.name for robot in results["robots"]],
            "scalars": scalars, "per-robot": per_robot, "series": series}


def save_metrics(data_dir, metrics):
    path = pathlib.Path(data_dir, METRICS_FILE)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=1)
    return path


def load_metrics(data_dir):
    with open(pathlib.Path(data_dir, METRICS_FILE)) as f:
        return json.load(f)
