"""Explicit assembly of the experiments used by the MRMR paper."""

import pathlib

from communication import PerfectCommunicationMedium
from exp_run_config import Config
from policy import RandomWaypointPolicy
from robot import Robot
from water_berry_farm import WBF_IM_DiskEstimator, WBF_Score_WeightedAsymmetric
from wbf_helper import create_wbfe, generate_fixed_budget_lawnmower, get_geometry
from wbf_simulate import save_simulation_results, simulate_1day

from .epmarket import EPM
from .mrmr_policies import MRMR_Contractor, MRMR_Pioneer


def run_mrmr_experiment(exp):
    """Construct and run one configured MRMR-paper comparison explicitly."""

    exp_env = Config().get_experiment(
        exp["exp_environment"], exp["run_environment"])
    exp_estimator = Config().get_experiment(
        exp["exp_estimator"], exp["run_estimator"])
    exp_score = Config().get_experiment(exp["exp_score"], exp["run_score"])

    geometry = get_geometry(exp_env["typename"])
    farm, environment = create_wbfe(exp_env)
    environment.proceed(exp["time-start-environment"])

    estimator = WBF_IM_DiskEstimator(
        geometry["width"], geometry["height"])
    estimator.name = exp_estimator["estimator-name"]
    evaluator = WBF_Score_WeightedAsymmetric()
    evaluator.name = exp_score["score-name"]

    EPM().reset()
    communication = PerfectCommunicationMedium(environment)
    robots = []
    for values in exp["robots"]:
        exp_policy = Config().get_experiment(
            values["exp-policy"], values["run-policy"])
        if "exp-policy-extra-parameters" in values:
            for key, value in values["exp-policy-extra-parameters"].items():
                exp_policy[key] = value

        if values["run-policy"].startswith("random-waypoint"):
            policy = RandomWaypointPolicy(
                1,
                [geometry["xmin"], geometry["ymin"]],
                [geometry["xmax"], geometry["ymax"]],
                exp_policy["seed"],
            )
        elif values["run-policy"] == "fixed-budget-lawnmower":
            policy = generate_fixed_budget_lawnmower(exp_policy, exp_env)
        elif values["name"] == "pio":
            policy = MRMR_Pioneer(exp_policy, exp_env)
        elif values["name"].startswith("con-"):
            policy = MRMR_Contractor(exp_policy, exp_env)
        else:
            raise Exception(
                f"No explicit MRMR policy construction for {values['name']}")

        policy.name = exp_policy["policy-name"]
        robot = Robot(values["name"], 0, 0, 0)
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
        communication_rounds=0,
    )
    results.update({
        "wbf": farm,
        "wbfe": environment,
        "estimator-name": estimator.name,
        "score-name": evaluator.name,
        "results-basedir": exp["data_dir"],
    })
    save_simulation_results(
        pathlib.Path(exp.data_dir(), "results.pickle"), results)
    exp.done()
    return results
