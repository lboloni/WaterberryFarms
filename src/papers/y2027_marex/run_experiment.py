"""Explicit assembly of the experiments used by the MAREX paper."""

import pathlib
import time

from communication import PerfectCommunicationMedium, traffic_summary
from exp_run_config import Config
from policy import RandomWaypointPolicy
from robot import Robot
from water_berry_farm import WBF_Score_VoI, WBF_Score_WeightedAsymmetric
from wbf_helper import create_estimator, create_wbfe, get_geometry
from wbf_simulate import save_simulation_results, simulate_1day

from exprunflow.metrics import save_metrics

from .balanced_areas import BALANCED_POLICIES, allocate_balanced_strips, create_balanced_policy
from .marex_metrics import LABEL_KEYS, collect_metrics
from .marex_policies import MAREX_Contractor, MAREX_Pioneer, MAREX_Policy


# the start of every robot, the home base
START = (0, 0)


def balanced_allocation(exp, geometry):
    """The balanced areas of the robots of the run, if it declares balanced-areas: the area the team divides 
    (default: the field), and the strips of its robots (default: all robots, nearest strip first), from 
    their common budget (default: the timesteps of the day). None if the run has no balanced areas."""
    config = exp.get("balanced-areas")
    if config is None:
        return None
    area = config.get("area") or [geometry["xmin"], geometry["ymin"], geometry["xmax"], geometry["ymax"]]
    names = config.get("robots") or [values["name"] for values in exp["robots"]]
    budget = config.get("budget") or exp["timesteps-per-day"]
    strips = allocate_balanced_strips([START] * len(names), area, geometry["velocity"], budget)
    return {"area": area, "budget": budget, "robots": dict(zip(names, strips))}


def run_marex_experiment(exp):
    """Construct and run one configured MAREX-paper comparison explicitly. Saves the results pickle and 
    the metrics (metrics.json) into the data directory of the exp/run."""

    start = time.time()
    exp_env = Config().get_experiment(
        exp["exp_environment"], exp["run_environment"])
    exp_estimator = Config().get_experiment(
        exp["exp_estimator"], exp["run_estimator"])
    exp_score = Config().get_experiment(exp["exp_score"], exp["run_score"])

    geometry = get_geometry(exp_env["typename"])
    farm, environment = create_wbfe(exp_env)
    environment.proceed(exp["time-start-environment"])

    estimator = create_estimator(exp_estimator, geometry)
    if exp["run_score"] == "voi":
        evaluator = WBF_Score_VoI(exp_score["v-pos"], exp_score["v-neg"], exp_score["v-unknown"])
    else:
        evaluator = WBF_Score_WeightedAsymmetric()
    evaluator.name = exp_score["score-name"]

    communication = PerfectCommunicationMedium(environment)
    balanced = balanced_allocation(exp, geometry)
    robots = []
    for values in exp["robots"]:
        exp_policy = Config().get_experiment(
            values["exp-policy"], values["run-policy"])
        if "exp-policy-extra-parameters" in values:
            for key, value in values["exp-policy-extra-parameters"].items():
                exp_policy[key] = value

        if values["run-policy"] in BALANCED_POLICIES:
            area = balanced["area"]
            estimator_of_robot = None
            if values["run-policy"] == "balanced-confidence-guided":
                estimator_of_robot = create_estimator(exp_estimator, geometry)
            policy = create_balanced_policy(values["run-policy"], exp_policy, balanced["robots"][values["name"]],
                                            area, geometry["velocity"], estimator_of_robot)
        elif values["run-policy"].startswith("random-waypoint"):
            policy = RandomWaypointPolicy(
                1,
                [geometry["xmin"], geometry["ymin"]],
                [geometry["xmax"], geometry["ymax"]],
                exp_policy["seed"],
            )
        elif values["name"] == "pio":
            policy = MAREX_Pioneer(exp_policy, exp_env)
        elif values["name"].startswith("con-"):
            policy = MAREX_Contractor(exp_policy, exp_env)
        else:
            raise Exception(
                f"No explicit MAREX policy construction for {values['name']}")

        policy.name = exp_policy["policy-name"]
        robot = Robot(values["name"], *START, 0)
        robot.assign_policy(policy)
        communication.add_robot(robot)
        robots.append(robot)

    results = simulate_1day(
        environment=environment,
        robots=robots,
        estimator=estimator,
        evaluator=evaluator,
        timesteps=exp["timesteps-per-day"],
        estimator_interval=exp["im_resolution"],
        communication=communication,
        communication_rounds=exp["communication-rounds"],
    )
    results.update({
        "wbf": farm,
        "wbfe": environment,
        "estimator-name": estimator.name,
        "score-name": evaluator.name,
        "results-basedir": exp["data_dir"],
        # the bandwidth used by the communication (DESIGN-COMMUNICATION.md, Section 2.1)
        "communication-traffic": communication.traffic,
        "communication-summary": traffic_summary(
            communication.traffic, exp["timesteps-per-day"]),
        # the strips of the robots exploring balanced areas (DESIGN-BALANCED-EXPLORATION-AREAS.md)
        "balanced-areas": balanced,
        # the robot-local maps of the explored cells of the MAREX robots
        "explored-maps": {robot.name: robot.policy.explored.cells for robot in robots
                          if isinstance(robot.policy, MAREX_Policy)},
    })
    save_simulation_results(
        pathlib.Path(exp.data_dir(), "results.pickle"), results)
    scalars, series, entities = collect_metrics(results, computation_seconds=time.time() - start)
    save_metrics(exp, scalars, series=series, entities=entities, label_keys=LABEL_KEYS)
    exp.done()
    return results
