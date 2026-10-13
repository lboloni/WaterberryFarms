"""
thompson_sampling.py

Adapted GUTS: decentralized Thompson sampling, a baseline of the MAREX paper after GUTS (Bakshi, Gupta,
Ghods, Schneider, "GUTS: Generalized Uncertainty-Aware Thompson Sampling for Multi-Agent Active Search",
ICRA 2023); see paper-MRMR, PLAN-ComparisonStudy.md, "Adapted GUTS". It keeps the coordination principle of
GUTS (asynchronous decisions, a Thompson sample from a private posterior, shared observations but no shared
plans) and adapts its posterior, reward and candidates to the noise-free single-cell sensing of WBF. Two
variants: A-GUTS (footprint 0, independent cells as in GUTS) and A-GUTS-SP (a spatial posterior).

Every robot keeps its own posterior of the diseased cells, from its own observations and those its
teammates broadcast (TeamKnowledge). Whenever it has no waypoint, it draws one Thompson sample of the
diseased cells from the posterior, and goes to the candidate waypoint whose path observes the most sampled
diseased cells per timestep of travel (in WBF the budget is counted in timesteps, so this rate replaces the
travel penalty of the GUTS reference code). The robots share observations but not plans: the randomness of the
samples spreads them. With footprint 0 the posterior is independent per cell, as in GUTS; with a positive
footprint a detection raises the probability of its neighbors, which can exploit clustered outbreaks.
"""

import numpy as np

from policy import AbstractWaypointPolicy

from .team_knowledge import ObservationSharing, TeamKnowledge, path_cells


class ThompsonSamplingPolicy(ObservationSharing, AbstractWaypointPolicy):
    """The DTS policy of one robot. exp_policy provides the budget (timesteps), the seed, the parameters
    prior, p-hit, p-false-alarm and footprint of the posterior, and candidates (the number of waypoints
    scored at a decision)."""

    def __init__(self, exp_policy, geometry):
        super().__init__()
        self.vel = geometry["velocity"]
        self.width, self.height = geometry["width"], geometry["height"]
        self.budget = exp_policy["budget"]
        self.candidates = exp_policy["candidates"]
        self.random = np.random.default_rng(exp_policy["seed"])
        self.knowledge = TeamKnowledge(self.width, self.height, prior=exp_policy["prior"],
                                       p_hit=exp_policy["p-hit"], p_false_alarm=exp_policy["p-false-alarm"],
                                       footprint=exp_policy["footprint"])
        self.waypoint = None
        self.used = 0  # the timesteps of the budget used
        self.decisions = 0

    def add_observation(self, obs):
        self.remember_observation(obs)

    def act(self, delta_t):
        self.used += delta_t
        if self.waypoint is None:
            self.waypoint = self.choose_waypoint()
            if self.waypoint is None:
                self.robot.add_action("vel [0, 0]")
                return
        if self.move_towards_location(self.waypoint[0], self.waypoint[1], self.vel, delta_t):
            self.waypoint = None

    def choose_waypoint(self):
        """Draw a Thompson sample of the diseased cells and return the best candidate waypoint, or None if no
        unobserved cell is within the remaining budget. The candidates are drawn half uniformly from the
        reachable unobserved cells, half from those that the sample says are diseased (the targets the sample
        proposes). A candidate is scored by the sampled diseased cells its path observes per timestep of
        travel, as the budget is counted in timesteps; ties go to the nearer candidate."""
        self.decisions += 1
        position = np.array([self.robot.x, self.robot.y])
        remaining = (self.budget - self.used) * self.vel
        unobserved = ~self.knowledge.observed
        if remaining < 1 or not unobserved.any():
            return None
        # the Thompson sample: every unobserved cell is diseased with its posterior probability
        sample = (self.random.random((self.width, self.height)) < self.knowledge.probability()) & unobserved
        xs, ys = np.nonzero(unobserved)
        distances = np.hypot(xs - position[0], ys - position[1])
        reachable = (distances <= remaining) & (distances > 0)
        if not reachable.any():
            return None
        uniform = np.nonzero(reachable)[0]
        targets = np.nonzero(reachable & sample[xs, ys])[0]
        half = self.candidates // 2
        chosen = self.random.choice(uniform, size=min(self.candidates - min(half, len(targets)), len(uniform)), replace=False)
        if len(targets):
            chosen = np.concatenate([self.random.choice(targets, size=min(half, len(targets)), replace=False), chosen])
        best, best_key = None, None
        for index in chosen:
            target = (int(xs[index]), int(ys[index]))
            path = path_cells(position, target, self.vel)
            cells = np.unique(path, axis=0)
            found = int(np.count_nonzero(sample[cells[:, 0], cells[:, 1]]))
            # the highest rate of sampled diseased cells per timestep; ties to the nearer candidate
            key = (found / len(path), -distances[index])
            if best_key is None or key > best_key:
                best, best_key = target, key
        return best
