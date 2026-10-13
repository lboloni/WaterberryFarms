"""
region_task_allocation.py

Region-based task allocation with a correlation kernel (RBTA), a baseline of the MAREX paper after Edmonds and
Yi, "Efficient Multi-Robot Inspection of Row Crops via Kernel Estimation and Region-Based Task Allocation",
ICRA 2021 (as restated in Chapter 4 of Edmonds' 2022 Rutgers dissertation); see paper-MRMR,
PLAN-ComparisonStudy.md, Plan C.

The field is partitioned among the robots by travel time plus remaining task time (the augmented Voronoi
partition). When a robot is free, it takes the unobserved cell of its region that minimizes
gamma_P distance + C, where C is the redundancy field, the correlation kernel stamped at every observed cell:
the cell least correlated with what has been sampled, discounted by distance. The method does not use the
measured values.

The adaptations to WBF (documented in the plan): a grid without rows (Euclidean travel time), a stationary
kernel estimated as the spatial autocorrelation of generated disease maps (with seeds outside the
replications, at the start of every run, as the generation is fast) instead of a neural kernel learned from
multi-season data, the redundancy over all the observed cells (WBF observes along the path), no stop to scan,
and decentralized computation of the same partition from the shared observations and task messages.
"""

import math

import numpy as np

from communication import Message
from disease_maps import generate_disease_map
from information_model import DISEASED
from policy import AbstractWaypointPolicy

from .team_knowledge import ObservationSharing, TeamKnowledge

_KERNELS = {}  # the estimated kernels, by (version, width, height, fraction, seeds, mask digest)


def estimate_kernel(width, height, version, fraction, seeds, immunity_mask=None, max_offset=30, threshold=0.05):
    """The stationary correlation kernel of the disease indicator of generated maps: the autocorrelation of
    the indicator (on the plantable cells), normalized by the variance and the overlap, averaged over the
    maps of the seeds, symmetric, 1 at the zero offset, truncated at the radius where it falls below the
    threshold. Returns a (2r+1, 2r+1) array."""
    key = (version, width, height, fraction, tuple(seeds),
           None if immunity_mask is None else hash(np.asarray(immunity_mask).tobytes()))
    if key in _KERNELS:
        return _KERNELS[key]
    mask = np.ones((width, height)) if immunity_mask is None else (np.asarray(immunity_mask) != -2) * 1.0
    shape = (2 * width, 2 * height)
    overlap = np.real(np.fft.ifft2(np.abs(np.fft.fft2(mask, shape)) ** 2))
    total = np.zeros(shape)
    for seed in seeds:
        model, _ = generate_disease_map(width, height, version, seed, fraction, immunity_mask=immunity_mask)
        indicator = (model.value < DISEASED) * 1.0
        mean = indicator[mask > 0].mean()
        variance = indicator[mask > 0].var()
        centered = (indicator - mean) * mask
        autocorrelation = np.real(np.fft.ifft2(np.abs(np.fft.fft2(centered, shape)) ** 2))
        total += autocorrelation / np.maximum(overlap, 1) / max(variance, 1e-12)
    correlation = np.fft.fftshift(total / len(seeds))
    cx, cy = width, height  # the zero offset after fftshift
    window = correlation[cx - max_offset:cx + max_offset + 1, cy - max_offset:cy + max_offset + 1]
    window = (window + window[::-1, ::-1]) / 2
    window[max_offset, max_offset] = 1.0
    dx, dy = np.meshgrid(np.arange(-max_offset, max_offset + 1), np.arange(-max_offset, max_offset + 1), indexing="ij")
    distance = np.hypot(dx, dy)
    above = distance[window >= threshold]
    radius = int(min(max_offset, math.ceil(above.max()))) if len(above) else 0
    kernel = np.where(distance <= radius, np.maximum(window, 0), 0)
    kernel = kernel[max_offset - radius:max_offset + radius + 1, max_offset - radius:max_offset + radius + 1]
    _KERNELS[key] = kernel
    return kernel


def partition(positions, remaining_times, width, height):
    """The augmented Voronoi partition: every cell to the robot with the smallest travel time plus remaining
    task time (speed 1); ties to the earlier robot of the order of the names. Returns (names, owner array)."""
    names = sorted(positions)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    times = np.stack([np.hypot(xs - positions[n][0], ys - positions[n][1]) + remaining_times[n] for n in names])
    return names, np.argmin(times, axis=0)


class RegionTaskAllocationPolicy(ObservationSharing, AbstractWaypointPolicy):
    """The RBTA policy of one robot"""

    def __init__(self, exp_policy, geometry, name, kernel):
        super().__init__()
        self.vel = geometry["velocity"]
        self.width, self.height = geometry["width"], geometry["height"]
        self.name_of_robot = name
        self.budget = exp_policy["budget"]
        self.gamma = exp_policy["gamma-p"]
        self.redundancy_of = exp_policy["redundancy"]  # "observed" or "targets" (published)
        self.knowledge = TeamKnowledge(self.width, self.height, footprint=0)
        self.kernel = kernel
        self.radius = kernel.shape[0] // 2
        self.redundancy = np.zeros((self.width, self.height))
        self.positions = {}  # name -> the cell of its last observation (its position)
        self.tasks = {}  # name -> {"target", "eta", "time"} from its last task message
        self.xs, self.ys = np.meshgrid(np.arange(self.width), np.arange(self.height), indexing="ij")
        self.target = None
        self.used = 0
        self.to_announce = None

    def stamp(self, x, y):
        r = self.radius
        x0, x1 = max(0, x - r), min(self.width, x + r + 1)
        y0, y1 = max(0, y - r), min(self.height, y + r + 1)
        self.redundancy[x0:x1, y0:y1] += self.kernel[x0 - x + r:x1 - x + r, y0 - y + r:y1 - y + r]

    def note_observed(self, x, y, diseased):
        if self.knowledge.add(x, y, diseased) and self.redundancy_of == "observed":
            self.stamp(x, y)

    def add_observation(self, obs):
        x, y = int(obs["x"]), int(obs["y"])
        self.note_observed(x, y, obs["TYLCV"]["value"] < DISEASED)
        self.to_share = {"type": "observation", "x": x, "y": y, "d": int(obs["TYLCV"]["value"] < DISEASED)}

    def act_send(self, round):
        super().act_send(round)
        if round == 0 and self.to_announce is not None:
            self.robot.com.send(self.robot, None, Message(self.to_announce))
            self.to_announce = None

    def act_receive(self, round, messages):
        for message in messages:
            content = message.content
            if not isinstance(content, dict):
                continue
            if content.get("type") == "observation":
                self.note_observed(content["x"], content["y"], bool(content["d"]))
                self.positions[message.sender_name] = (content["x"], content["y"])
            elif content.get("type") == "task":
                self.tasks[message.sender_name] = {"target": (content["tx"], content["ty"]),
                                                   "eta": content["eta"], "time": self.used}
                if self.redundancy_of == "targets":
                    self.stamp(content["tx"], content["ty"])

    def allocate(self, name, position, anchors, redundancy, observed):
        """The task of a free robot at position: the cell of its region of the augmented Voronoi partition of
        the anchors (name -> (point, remaining time)) that minimizes gamma_P distance + redundancy, among
        the unobserved cells within the budget; if its region has none, the best such cell anywhere"""
        names, owner = partition({n: a[0] for n, a in anchors.items()},
                                 {n: a[1] for n, a in anchors.items()}, self.width, self.height)
        distance = np.hypot(self.xs - position[0], self.ys - position[1])
        budget = (self.budget - self.used) * self.vel
        feasible = ~observed & (distance <= budget) & (distance > 0)
        region = feasible & (owner == names.index(name))
        candidates = region if region.any() else feasible
        if not candidates.any():
            return None, 0.0
        score = np.where(candidates, self.gamma * distance + redundancy, np.inf)
        best = np.flatnonzero(score == score.min())
        nearest = best[np.argmin(distance.flat[best])]
        tx, ty = np.unravel_index(nearest, score.shape)
        return (int(tx), int(ty)), float(distance[tx, ty]) / self.vel

    def choose_task(self):
        """The robot's next task. As the central coordinator of the paper allocates the free robots one after
        the other, a free robot first replays, in the order of the names, the allocations of the free
        teammates before it: they have the same knowledge (the observations up to the previous timestep), so
        they compute the same allocations. Busy teammates are anchored at their task points, with their
        remaining task time (the augmented Voronoi partition)."""
        me = self.name_of_robot
        self.positions[me] = (int(self.robot.x), int(self.robot.y))
        anchors, free = {}, []
        for name, position in self.positions.items():
            task = self.tasks.get(name)
            remaining = 0.0 if task is None else task["eta"] - (self.used - task["time"])
            if name != me and remaining > 0:
                anchors[name] = (task["target"], remaining)
            else:
                anchors[name] = (position, 0.0)
                free.append(name)
        redundancy = self.redundancy.copy()
        observed = self.knowledge.observed.copy()
        for name in sorted(free):
            target, eta = self.allocate(name, self.positions[name], anchors, redundancy, observed)
            if name == me:
                if target is not None:
                    self.to_announce = {"type": "task", "tx": target[0], "ty": target[1], "eta": round(eta, 1)}
                    if self.redundancy_of == "targets":
                        self.stamp(*target)
                return target
            if target is not None:
                # the teammate's task: its point is taken, and the teammate is anchored there
                anchors[name] = (target, eta)
                observed[target] = True
                r = self.radius
                x, y = target
                x0, x1 = max(0, x - r), min(self.width, x + r + 1)
                y0, y1 = max(0, y - r), min(self.height, y + r + 1)
                redundancy[x0:x1, y0:y1] += self.kernel[x0 - x + r:x1 - x + r, y0 - y + r:y1 - y + r]
        return None

    def act(self, delta_t):
        self.used += delta_t
        if self.used <= delta_t:
            # the first timestep: the robot learns its teammates from their first observation broadcast
            self.robot.add_action("vel [0, 0]")
            return
        if self.target is None:
            self.target = self.choose_task()
            if self.target is None:
                self.robot.add_action("vel [0, 0]")
                return
        if self.move_towards_location(self.target[0], self.target[1], self.vel, delta_t):
            self.target = None
