"""
ucb_auction.py

Adapted UCB auction (A-UCBA), a baseline of the MAREX paper after Jia, Liu, Lu, Zhang, Su, "Uncertain-aware
Informative Task Planning and Assignment for Multiple-UUVs Cooperative Underwater Exploration", IROS 2025, as
implemented in the authors' code (github.com/Chengfeng-Jia/UUVs_explore, main_code/submit.ipynb); see
paper-MRMR, PLAN-ComparisonStudy.md, Plan B.

A central auctioneer keeps a sparse Gaussian process (GPy) of the observations of the team. Every round, it
selects the interest points, the top points of the upper confidence bound mu + kappa sigma at a minimum
separation, and auctions them to the idle robots, with the published cost 10 d - 2 sigma + UCB + 5 (0.5 d):
the interest points in UCB order, each to its lowest-cost idle bidder. A robot moves straight to its point,
observing along the way, and is idle again on arrival. In WBF one robot of the team is also the auctioneer;
the observations (with the idle flag) and the assignments are messages.

The adaptations to WBF (documented in the plan): no prior map, many targets and no stop at the first
detection, observed cells excluded from the interest points, only the active assignments excluded around new
ones, the budget in timesteps (the published cost only ranks the bids), and the GP data updated every
refit-interval timesteps and re-optimized every optimize-interval timesteps.
"""

import math

import numpy as np

from communication import Message
from policy import AbstractWaypointPolicy

from .team_knowledge import ObservationSharing, TeamKnowledge

# GPy is imported lazily, so that the module (and the metrics) can be imported without it


class UCBAuctionGP:
    """The auctioneer's model: a GPy sparse GP of the observed cells, 1 for diseased and the healthy value
    otherwise; without observations, the prior (mean 0, variance 1)"""

    def __init__(self, width, height, lengthscale, inducing, healthy_value):
        self.width, self.height = width, height
        self.lengthscale, self.healthy_value = lengthscale, healthy_value
        xs = np.linspace(0, width - 1, inducing)
        ys = np.linspace(0, height - 1, inducing)
        self.inducing = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        self.grid = np.stack(np.meshgrid(np.arange(width), np.arange(height), indexing="ij"), -1).reshape(-1, 2).astype(float)
        self.model = None
        self.mu = np.zeros((width, height))
        self.sigma = np.ones((width, height))

    def refit(self, knowledge, optimize):
        """Update the model with the observed cells of the knowledge; optimize the hyperparameters and the
        inducing points if asked; predict mu and sigma on the grid"""
        import GPy
        xs, ys = np.nonzero(knowledge.observed)
        if len(xs) == 0:
            return
        X = np.stack([xs, ys], -1).astype(float)
        Y = np.where(knowledge.diseased[xs, ys], 1.0, self.healthy_value)[:, None]
        if self.model is None:
            kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=self.lengthscale)
            self.model = GPy.models.SparseGPRegression(X, Y, kernel, Z=self.inducing.copy())
            optimize = True
        else:
            self.model.set_XY(X, Y)
        if optimize:
            self.model.optimize(messages=False, max_iters=100)
        mu, var = [], []
        for chunk in np.array_split(self.grid, max(1, len(self.grid) // 10000)):
            m, v = self.model.predict(chunk)
            mu.append(m[:, 0])
            var.append(v[:, 0])
        self.mu = np.concatenate(mu).reshape(self.width, self.height)
        self.sigma = np.sqrt(np.maximum(np.concatenate(var), 0)).reshape(self.width, self.height)


def interest_points(ucb, available, count, separation, rng):
    """The top count cells of the UCB among the available ones, at least separation apart (published:
    find_global_interest_points); ties broken by a seeded random order"""
    xs, ys = np.nonzero(available)
    if len(xs) == 0:
        return []
    values = ucb[xs, ys]
    order = np.lexsort((rng.random(len(xs)), -values))
    selected = []
    for index in order:
        point = (int(xs[index]), int(ys[index]))
        if all(math.dist(point, p) >= separation for p in selected):
            selected.append(point)
            if len(selected) == count:
                break
    return selected


def auction(points, mu_sigma_ucb, idle, positions, remaining, active, separation, weights):
    """The published auction (auction_points): the interest points in UCB order, skipping those within the
    separation of an active assignment; bids from the idle robots whose remaining budget covers the travel,
    with cost w_d d - w_s sigma + w_u UCB + w_e 0.5 d; every point to its lowest-cost bidder that has not won
    yet. Returns {robot: point}"""
    w_d, w_s, w_u, w_e = weights
    assignments = {}
    for point in points:
        if any(math.dist(point, a) < separation for a in active):
            continue
        sigma, ucb = mu_sigma_ucb(point)
        bids = []
        for name in sorted(idle):
            if name in assignments:
                continue
            distance = math.dist(positions[name], point)
            if distance > remaining:
                continue
            cost = w_d * distance - w_s * sigma + w_u * ucb + w_e * 0.5 * distance
            bids.append((cost, name))
        if bids:
            cost, winner = min(bids)
            assignments[winner] = point
            active = list(active) + [point]
    return assignments


class UCBAuctionPolicy(ObservationSharing, AbstractWaypointPolicy):
    """The A-UCBA policy of one robot; the robot with auctioneer: true also runs the auction"""

    def __init__(self, exp_policy, geometry, name):
        super().__init__()
        self.vel = geometry["velocity"]
        self.width, self.height = geometry["width"], geometry["height"]
        self.name_of_robot = name
        self.budget = exp_policy["budget"]
        self.knowledge = TeamKnowledge(self.width, self.height, footprint=0)
        self.target = None
        self.used = 0
        self.auctioneer = bool(exp_policy.get("auctioneer", False))
        if self.auctioneer:
            self.kappa = exp_policy["kappa"]
            self.count = exp_policy["interest-points"]
            self.separation = exp_policy["separation"]
            self.weights = tuple(exp_policy["cost-weights"])
            self.refit_interval = exp_policy["refit-interval"]
            self.optimize_interval = exp_policy["optimize-interval"]
            inducing = exp_policy["inducing-per-side-100"] if self.width <= 100 else exp_policy["inducing-per-side-200"]
            self.gp = UCBAuctionGP(self.width, self.height, exp_policy["lengthscale"], inducing, exp_policy["healthy-value"])
            self.random = np.random.default_rng(exp_policy["seed"])
            self.team = {}  # robot name -> {"position": (x, y), "idle": bool}
            self.active = {}  # robot name -> its assigned point, while it travels there
            self.assignments_made = 0

    # the observations, extended with the idle flag and shared in round 0 (ObservationSharing)
    def add_observation(self, obs):
        self.remember_observation(obs)
        if self.auctioneer:
            self.team[self.name_of_robot] = {"position": (int(obs["x"]), int(obs["y"])), "idle": self.target is None}

    def act_send(self, round):
        if round == 0 and getattr(self, "to_share", None) is not None:
            self.to_share["idle"] = int(self.target is None)
        super().act_send(round)
        if round == 1 and self.auctioneer:
            for name, point in self.run_auction().items():
                if name == self.name_of_robot:
                    self.target = point
                else:
                    self.robot.com.send(self.robot, name, Message({"type": "assign", "x": point[0], "y": point[1]}))

    def act_receive(self, round, messages):
        super().act_receive(round, messages)
        for message in messages:
            content = message.content
            if not isinstance(content, dict):
                continue
            if content.get("type") == "observation" and self.auctioneer:
                self.team[message.sender_name] = {"position": (content["x"], content["y"]), "idle": bool(content.get("idle", 1))}
            elif content.get("type") == "assign":
                self.target = (content["x"], content["y"])

    def run_auction(self):
        """The auctioneer's round: refit the model on schedule, then auction the interest points to the idle
        robots"""
        timestep = int(self.used)
        if timestep % self.refit_interval == 0:
            self.gp.refit(self.knowledge, optimize=timestep % self.optimize_interval == 0)
        for name, state in self.team.items():
            if state["idle"]:
                self.active.pop(name, None)
        idle = {name for name, state in self.team.items() if state["idle"]}
        if not idle:
            return {}
        ucb = self.gp.mu + self.kappa * self.gp.sigma
        points = interest_points(ucb, ~self.knowledge.observed, self.count, self.separation, self.random)
        remaining = (self.budget - self.used) * self.vel
        positions = {name: state["position"] for name, state in self.team.items()}
        result = auction(points, lambda p: (self.gp.sigma[p], ucb[p]), idle, positions, remaining,
                         list(self.active.values()), self.separation, self.weights)
        for name, point in result.items():
            self.active[name] = point
            self.team[name]["idle"] = False
        self.assignments_made += len(result)
        return result

    def act(self, delta_t):
        self.used += delta_t
        if self.target is None:
            self.robot.add_action("vel [0, 0]")
            return
        if self.move_towards_location(self.target[0], self.target[1], self.vel, delta_t):
            self.target = None
