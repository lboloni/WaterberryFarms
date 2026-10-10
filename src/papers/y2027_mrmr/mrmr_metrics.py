"""
mrmr_metrics.py

The measurements of one MRMR run, saved as its metrics.json with exprunflow.metrics.save_metrics, from
which the replications are aggregated without loading their results pickles.
See DESIGN-MultiSeedEvaluation.md, Section 5, and ExpRunFlow's DESIGN-Replication.md, Section 3.
"""

import numpy as np

from policy import RandomWaypointPolicy
from water_berry_farm import voi_credits

from .mrmr_policies import MRMR_Contractor, MRMR_Pioneer

# the keys of a run that identify its cell (the labels of the aggregation)
LABEL_KEYS = ["map-size", "scenario", "approach"]
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


def collect_metrics(results, computation_seconds=None):
    """The measurements of a run: the final scalars, the series sampled at the scoring events (indexed by
    timestep), and the values per robot (entities, grouped by the robot's role)"""
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

    entities = {}
    credits = voi_credits(results, "voi-absolute") if scored_voi else None
    for robot in results["robots"]:
        name = robot.name
        entities[name] = {
            "group": robot_role(robot),
            "cells-observed": sum(1 for _, rec in cells if rec["first-robot"] == name),
            "diseased-found": sum(1 for _, rec in found if rec["first-robot"] == name),
            "bytes-transmitted": summary["per-sender"].get(name, {}).get("bytes-transmitted", 0),
        }
        if credits is not None:
            entities[name]["voi-absolute"] = float(credits[name].sum())

    if scored_voi:
        for variant in VOI_VARIANTS:
            scalars[variant] = float(results["score"][variant])
            series[variant] = [float(event["score"][variant]) for event in events]

    return scalars, series, entities
