"""
balanced_areas.py

Balanced exploration areas for the multi-robot baselines of the MAREX paper
(DESIGN-BALANCED-EXPLORATION-AREAS.md): the field is divided into one horizontal strip per robot such that
every robot, after travelling to its strip, could cover it with a lawnmower of the same pass spacing (the
lawnmower equivalent). The robots farther away get thinner strips. In its strip, a robot explores with a
lawnmower, random waypoints or confidence-guided exploration.
"""

import math

import numpy as np

from policy import AbstractWaypointPolicy, FollowPathPolicy, RandomWaypointPolicy
from papers.y2023_confidenceguided.confidence_guided_ipp_policy import ConfidenceGuidedPathPlanning

# the policies of the robots that explore a balanced area
BALANCED_POLICIES = ["balanced-lawnmower", "balanced-random-waypoint", "balanced-confidence-guided"]


def pass_cost(travel, passes, width, spacing):
    """The distance a lawnmower equivalent covers: the travel to the entry point, then the horizontal
    passes of the given width, spacing apart"""
    return travel + passes * width + (passes - 1) * spacing


def entry_point(start, x_min, x_max, y):
    """The end of the pass at y that is nearer to the start"""
    return min([(x_min, y), (x_max, y)], key=lambda point: math.dist(start, point))


def greedy_blocks(starts, area, distance, spacing):
    """The blocks of consecutive passes of a global lawnmower with the given spacing, assigned from the
    first robot to the last: every robot gets the most passes its distance budget allows after its travel"""
    x_min, y_min, x_max, _ = area
    width = x_max - x_min
    first, blocks = 0, []
    for start in starts:
        entry = entry_point(start, x_min, x_max, y_min + first * spacing)
        travel = math.dist(start, entry)
        # the tolerance keeps a pass that exactly uses up the budget
        passes = max(0, math.floor((distance - travel + spacing) / (width + spacing) + 1e-9))
        blocks.append({"first": first, "passes": passes, "entry": entry, "travel": travel})
        first += passes
    return blocks


def allocate_balanced_strips(starts, area, velocity, budget):
    """Divide the area [x_min, y_min, x_max, y_max] into one horizontal strip per robot, from the robots'
    starts, their velocity and their time budget. The strips are the blocks of consecutive passes of one
    global lawnmower over the area, with the smallest pass spacing at which the robots, every one after
    its travel, cover all the passes. Returns one dict per robot, in the order of the starts (nearest strip
    first): the area of its strip (None if it cannot afford a pass), its entry point, travel, the budget
    left after the travel, its first pass and number of passes, and the spacing and the y of its passes."""
    x_min, y_min, x_max, y_max = area
    height = y_max - y_min
    distance = velocity * budget
    if height <= 0 or x_max <= x_min:
        raise ValueError(f"Empty area {area}")
    def passes_for(count):
        return sum(block["passes"] for block in greedy_blocks(starts, area, distance, height / (count - 1)))
    if passes_for(2) < 2:
        raise ValueError(f"The robots cannot cover {area} with two passes in a budget of {budget}")
    # the most passes: P passes, spacing height / (P - 1), are affordable while the robots provide them
    count = 2
    while passes_for(count + 1) >= count + 1:
        count += 1
    spacing = height / (count - 1)
    allocation, left = [], count
    for block in greedy_blocks(starts, area, distance, spacing):
        passes = min(block["passes"], left)
        left -= passes
        ys = [y_min + (block["first"] + j) * spacing for j in range(passes)]
        strip = None
        if passes:
            strip = [x_min, max(y_min, ys[0] - spacing / 2), x_max, min(y_max, ys[-1] + spacing / 2)]
        allocation.append({"area": strip, "entry": list(block["entry"]), "travel": block["travel"],
                           "budget": budget - block["travel"] / velocity, "first": block["first"],
                           "passes": passes, "spacing": spacing, "ys": ys})
    return allocation


def lawnmower_passes(allocation, area):
    """The waypoints of the passes of a robot's strip, starting at its entry point and alternating ends"""
    x_min, _, x_max, _ = area
    near = allocation["entry"][0]
    far = x_max if near == x_min else x_min
    waypoints = []
    for j, y in enumerate(allocation["ys"]):
        begin, end = (near, far) if j % 2 == 0 else (far, near)
        waypoints += [[begin, y], [end, y]]
    return np.array(waypoints)


class TransitThenExplorePolicy(AbstractWaypointPolicy):
    """Fly straight to the entry point of the robot's area, then hand over to the explorer of the area"""
    def __init__(self, vel, entry, explorer):
        super().__init__()
        self.vel = vel
        self.entry = entry
        self.explorer = explorer
        self.arrived = False

    @property
    def robot(self):
        return self._robot

    @robot.setter
    def robot(self, robot):
        """The explorer acts for the same robot (Robot.assign_policy sets the robot of the policy directly)"""
        self._robot = robot
        self.explorer.robot = robot

    def act(self, delta_t):
        if self.arrived:
            self.explorer.act(delta_t)
        elif self.move_towards_location(self.entry[0], self.entry[1], self.vel, delta_t):
            self.arrived = True

    def add_observation(self, obs):
        self.explorer.add_observation(obs)


class AreaConfidenceGuided(ConfidenceGuidedPathPlanning):
    """Confidence-guided exploration in a rectangular area: the next waypoint is the cell of the highest
    uncertainty of the embedded estimator within the span around the robot, restricted to the area"""
    def __init__(self, vel, low_point, high_point, estimator, span=5, seed=0):
        super().__init__(vel, low_point, high_point, estimator, span=span, seed=seed)
        self.name = self.__class__.__name__

    def generate_feasible_waypoints(self):
        """The cells of the area within the span around the robot, except its current cell. If the robot
        is farther than the span from the area, the cells of the area nearest to it."""
        x, y = int(round(self.robot.x)), int(round(self.robot.y))
        low_x, low_y = self.low_point
        high_x, high_y = self.high_point
        cells = [(cx, cy) for cx in range(max(low_x, x - self.span), min(high_x, x + self.span) + 1)
                 for cy in range(max(low_y, y - self.span), min(high_y, y + self.span) + 1) if (cx, cy) != (x, y)]
        if not cells:
            cells = [(min(max(x, low_x), high_x), min(max(y, low_y), high_y))]
        return cells

    def choose_next_waypoint(self, feasible_waypoints):
        """The feasible waypoint of the highest uncertainty, ties broken randomly"""
        uncertainty = self.get_im().im_tylcv.uncertainty
        values = np.array([uncertainty[cx, cy] for cx, cy in feasible_waypoints])
        best = np.flatnonzero(values == values.max())
        return feasible_waypoints[self.random.choice(best)]


def create_balanced_policy(run_policy, exp_policy, allocation, area, velocity, estimator=None):
    """The policy of a robot exploring its balanced area: the transit to the entry point, then the explorer
    of the run-policy (BALANCED_POLICIES). The confidence-guided explorer needs its own estimator."""
    if allocation["area"] is None:
        raise ValueError(f"{exp_policy['policy-name']} got no area: the budget does not reach it")
    x_min, y_min, x_max, y_max = allocation["area"]
    if run_policy == "balanced-lawnmower":
        explorer = FollowPathPolicy(vel=velocity, waypoints=lawnmower_passes(allocation, area), repeat=False)
    elif run_policy == "balanced-random-waypoint":
        explorer = RandomWaypointPolicy(velocity, [x_min, y_min], [x_max, y_max], exp_policy["seed"])
    elif run_policy == "balanced-confidence-guided":
        explorer = AreaConfidenceGuided(velocity, [math.ceil(x_min), math.ceil(y_min)],
                                        [math.floor(x_max), math.floor(y_max)], estimator,
                                        span=exp_policy["span"], seed=exp_policy["seed"])
    else:
        raise ValueError(f"Unknown balanced policy {run_policy}")
    return TransitThenExplorePolicy(velocity, allocation["entry"], explorer)
