"""Simulation lifecycle and result persistence for Waterberry Farms."""

import gzip as compress
import logging
import pickle
import time


logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


class TimeTrack:
    """Track policy computation time and periodically report progress."""

    def __init__(self):
        self.start_time = time.time_ns()
        self.last_start_time = self.start_time

    def policy_start(self):
        self.policy_start_time = time.time_ns()

    def policy_finish(self, results):
        policy_finish_time = time.time_ns()
        results["computation-cost-policy"].append(
            policy_finish_time - self.policy_start_time)

    def current(self, timestep, timesteps):
        current_time = time.time_ns()
        if current_time - self.last_start_time > 10e9:
            elapsed = int((current_time - self.start_time) / 1e9)
            print(f"At {timestep} / {timesteps} elapsed {elapsed} seconds")
            self.last_start_time = current_time


def simulate_1day(*, environment, robots, estimator, evaluator, timesteps,
                  estimator_interval, communication=None,
                  communication_rounds=0, after_timestep=None,
                  after_day=None):
    """Run one day with caller-constructed simulation components."""

    robots = sorted(robots, key=lambda robot: robot.name)
    names = [robot.name for robot in robots]
    if len(names) != len(set(names)):
        raise Exception("Robot names must be unique")
    for robot in robots:
        if robot.policy is None:
            raise Exception(f"Robot {robot.name} has no policy")
        robot.im = estimator

    results = {
        "robots": robots,
        "robot-names": names,
        "score-events": [],
        "observations": [],
        "positions": [],
        "computation-cost-policy": [],
        "simulation-timestep": 0,
    }
    results["scores"] = results["score-events"]

    environment_time = environment.time
    interval_count = 0
    time_track = TimeTrack()
    for timestep in range(int(timesteps)):
        interval_count = _simulate_timestep(
            results=results,
            timestep=timestep,
            timesteps=timesteps,
            estimator_interval=estimator_interval,
            interval_count=interval_count,
            environment=environment,
            environment_time=environment_time,
            robots=robots,
            estimator=estimator,
            evaluator=evaluator,
            communication=communication,
            communication_rounds=communication_rounds,
            after_timestep=after_timestep,
            time_track=time_track,
        )

    if after_day is not None:
        after_day(results, environment, robots, estimator, evaluator)
    if environment.time != environment_time:
        raise Exception("Environment time changed during a one-day simulation")
    return results


def _simulate_timestep(*, results, timestep, timesteps, estimator_interval,
                       interval_count, environment, environment_time, robots,
                       estimator, evaluator, communication,
                       communication_rounds, after_timestep, time_track):
    """Run one canonical communication-to-hook simulation timestep."""

    if timestep != results["simulation-timestep"]:
        raise Exception("Simulation timesteps must be consecutive")
    if environment.time != environment_time:
        raise Exception("Environment time changed during a one-day simulation")

    time_track.policy_start()
    for round_number in range(communication_rounds):
        for robot in robots:
            robot.policy.act_send(round_number)
        for robot in robots:
            if communication.robots[robot.name] is not robot:
                raise Exception(f"Mailbox {robot.name} is not owned by its robot")
            messages = communication.receive(robot)
            robot.policy.act_receive(round_number, messages)

    for robot in robots:
        robot.enact_policy()
    for robot in robots:
        robot.proceed(1)

    positions = [
        [int(robot.x), int(robot.y), timestep]
        for robot in robots
    ]
    observations = [
        environment.get_observation(position)
        for position in positions
    ]
    if len(observations) != len(robots):
        raise Exception("Every robot must produce one observation per timestep")

    for observation in observations:
        estimator.add_observation(observation)
    for robot, observation in zip(robots, observations):
        robot.add_observation(observation)

    results["positions"].append(positions)
    results["observations"].append(observations)

    time_track.policy_finish(results)
    interval_count += 1
    if interval_count == estimator_interval or timestep + 1 == timesteps:
        estimator.proceed(interval_count)
        results["score"] = evaluator.score(environment, estimator)
        if (results["score-events"] and
                results["score-events"][-1]["timestep"] >= timestep):
            raise Exception("Score event timestamps must be strictly increasing")
        results["score-events"].append({
            "timestep": timestep,
            "score": results["score"],
        })
        interval_count = 0

    results["simulation-timestep"] = timestep + 1
    if after_timestep is not None:
        after_timestep(results, environment, robots, estimator, evaluator)
    if environment.time != environment_time:
        raise Exception("Environment time changed during a one-day simulation")
    time_track.current(timestep, timesteps)
    return interval_count


def save_simulation_results(resultsfile, results):
    """Save a caller-selected result dictionary as a compressed pickle."""

    print(f"Saving results to: {resultsfile}")
    with compress.open(resultsfile, "wb") as output:
        pickle.dump(results, output)
