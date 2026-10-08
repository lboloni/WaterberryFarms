"""Run Waterberry Farms with a policy and evaluator defined by the caller."""

import tempfile

from robot import Robot
from water_berry_farm import (
    MiniberryFarm,
    WaterberryFarmEnvironment,
    WBF_IM_DiskEstimator,
)
from wbf_simulate import simulate_1day


class ExternalEastwardPolicy:
    def __init__(self):
        self.name = "external-eastward"
        self.observations = []

    def act(self, delta_t):
        self.robot.add_action("vel [1, 0]")

    def add_observation(self, observation):
        self.observations.append(observation)


class ExternalObservationCountEvaluator:
    def score(self, environment, estimator):
        return len(estimator.im_tylcv.observations)


def main():
    with tempfile.TemporaryDirectory() as directory:
        farm = MiniberryFarm(scale=1)
        farm.create_type_map()
        environment = WaterberryFarmEnvironment(
            farm, use_saved=False, seed=10, savedir=directory)
        estimator = WBF_IM_DiskEstimator(environment.width, environment.height)

        robot = Robot("external-robot", 0, 0, 0)
        robot.assign_policy(ExternalEastwardPolicy())

        results = simulate_1day(
            environment=environment,
            robots=[robot],
            estimator=estimator,
            evaluator=ExternalObservationCountEvaluator(),
            timesteps=3,
            estimator_interval=1,
        )

        print(results["positions"])
        print(results["score-events"])


if __name__ == "__main__":
    main()
