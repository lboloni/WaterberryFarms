"""
team_knowledge.py

The common infrastructure of the reimplemented baselines of the MAREX paper (paper-MRMR,
PLAN-ComparisonStudy.md, "Common infrastructure"): what a robot knows about the field from its own
observations and those its teammates broadcast, the cells a straight move observes, and the sharing of
observations through the communication medium.
"""

import math

import numpy as np

from communication import Message
from information_model import DISEASED


def is_diseased(observation):
    """Whether an observation of the simulator shows a diseased plant (TYLCV value below DISEASED)"""
    return observation["TYLCV"]["value"] < DISEASED


def path_cells(start, end, vel=1.0):
    """The cells a robot observes when it moves straight from start to end, one step of vel per timestep,
    as AbstractWaypointPolicy.move_towards_location moves it and wbf_simulate records it: the positions
    start + k u (u the unit direction) while the target is farther than one step, then the target itself,
    every one truncated with int(). The start cell, already observed, is not included. Returns an (n, 2)
    integer array, possibly with repeated cells."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    distance = float(np.hypot(*delta))
    if distance == 0:
        return np.zeros((0, 2), dtype=int)
    steps = math.ceil(distance / vel) - 1  # the steps before the final jump onto the target
    k = np.arange(1, steps + 1)[:, None]
    points = start + k * (delta / distance) * vel
    points = np.vstack([points, end[None, :]])
    return points.astype(int)


class TeamKnowledge:
    """What a robot knows about the disease of the field: the cells observed by itself or by a teammate,
    their values, and a Bayesian log-odds grid of the probability that a cell is diseased. It is the model
    of information_model.OccupancyGridIM, updated incrementally: an observation adds its evidence to the
    cells around it, weighted by exp(-d^2 / footprint^2) (footprint 0: only the cell itself). The simulated
    sensor is deterministic, so a cell counts once, at its first observation."""

    def __init__(self, width, height, prior=0.05, p_hit=0.99, p_false_alarm=0.01, footprint=2.0):
        self.width, self.height = width, height
        self.footprint = footprint
        self.observed = np.zeros((width, height), dtype=bool)
        self.diseased = np.zeros((width, height), dtype=bool)
        self.logodds = np.full((width, height), math.log(prior / (1 - prior)))
        self.evidence_positive = math.log(p_hit / p_false_alarm)
        self.evidence_negative = math.log((1 - p_hit) / (1 - p_false_alarm))
        radius = int(math.ceil(3 * footprint))
        dx, dy = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1), indexing="ij")
        if footprint > 0:
            self.stamp = np.exp(-(dx**2 + dy**2) / footprint**2)
        else:
            self.stamp = (dx**2 + dy**2 == 0) * 1.0
        self.radius = radius

    def add(self, x, y, diseased):
        """Add the observation of a cell; returns False if the cell was already known"""
        if not (0 <= x < self.width and 0 <= y < self.height) or self.observed[x, y]:
            return False
        self.observed[x, y] = True
        self.diseased[x, y] = diseased
        evidence = self.evidence_positive if diseased else self.evidence_negative
        r = self.radius
        x0, x1 = max(0, x - r), min(self.width, x + r + 1)
        y0, y1 = max(0, y - r), min(self.height, y + r + 1)
        self.logodds[x0:x1, y0:y1] += evidence * self.stamp[x0 - x + r:x1 - x + r, y0 - y + r:y1 - y + r]
        return True

    def probability(self):
        """The probability that every cell is diseased: known for the observed cells, the log-odds grid for
        the others"""
        probability = 1.0 / (1.0 + np.exp(-np.clip(self.logodds, -30, 30)))
        probability[self.observed] = self.diseased[self.observed]
        return probability


class ObservationSharing:
    """A mixin for the policies that share every observation with their teammates: in round 0 of every
    timestep, the robot broadcasts the cell it observed, {"type": "observation", "x", "y", "d"} (d = 1 if
    diseased), and adds the observations it receives to its TeamKnowledge (self.knowledge). The policy calls
    remember_observation from its add_observation."""

    def remember_observation(self, observation):
        x, y, diseased = int(observation["x"]), int(observation["y"]), is_diseased(observation)
        self.knowledge.add(x, y, diseased)
        self.to_share = {"type": "observation", "x": x, "y": y, "d": int(diseased)}

    def act_send(self, round):
        if round == 0 and getattr(self, "to_share", None) is not None:
            self.robot.com.send(self.robot, None, Message(self.to_share))
            self.to_share = None

    def act_receive(self, round, messages):
        for message in messages:
            content = message.content
            if isinstance(content, dict) and content.get("type") == "observation":
                self.knowledge.add(content["x"], content["y"], bool(content["d"]))
