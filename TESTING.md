# Correctness Baseline

## Supported test surface

The deterministic correctness suite covers:

- Environment timing, epidemic spread, soil evaporation, and precalculated replay.
- Waterberry and Miniberry geometry.
- Robot actions and waypoint policies.
- Lawnmower, spiral, and GLR path generation.
- Point, adaptive-disk, and Gaussian-process information models.
- Weighted and asymmetric scoring.
- Experiment configuration assembly and output-directory creation modes.
- Unified one- and multi-robot simulation through caller-constructed components.
- Perfect communication and MRMR market primitives.
- Result serialization.

Notebooks, files under `obsolete`, LAIP, Bounomodes, and unfinished paper prototypes are not part of the supported test surface.

## Canonical simulation lifecycle

The keyword-only `simulate_1day` function implements the canonical lifecycle. Robots are ordered by their unique names, and each phase completes for all robots before the next phase begins:

1. Perform every communication round, with all sends before any receives.
2. Ask every policy to schedule actions.
3. Execute every robot's scheduled and every-step actions.
4. Record every post-movement position.
5. Acquire every observation from the unchanged environment snapshot.
6. Add all observations to the shared estimator, then deliver each observation to its originating policy.
7. Update the estimator and calculate a score event when `im_resolution` timesteps have elapsed or at the final partial interval.
8. Call `hook-after-timestep` after the timestep state is final.

`hook-after-day` is called once after the day's last timestep. Policies deciding at timestep `t` can use observations only through `t-1`; positions and observations stamped `t` describe the state after movement at `t`.

The simulator treats one robot timestep as one unit for policy and movement calls. The caller advances the environment to the desired starting time before the run. `Environment.time` must then remain constant throughout the robot timesteps and both hooks. No multi-day runner currently exists. A future multi-day runner must call the day hook before advancing the environment once at the boundary and must continue the absolute robot timestamp across days. `Environment.time_expansion` belongs only to environment evolution and does not alter robot timestamps.

## Canonical result schema

- `positions[t][r]` and `observations[t][r]` are always lists in canonical robot-name order, including when there is one robot.
- `robot-names` records that canonical order.
- Observation and position timestamps are zero-based absolute robot timesteps.
- `score-events` contains only estimator-update events in the form `{"timestep": t, "score": score}`. `scores` is an alias retained for result-field compatibility; it is not a dense per-timestep series.
- `simulation-timestep` is the number of completed robot timesteps.
- `computation-cost-policy` contains one elapsed-time measurement per completed timestep.

## External component contract

Tests define policies, estimators, evaluators, environments, and robots outside the framework classes and pass them directly to `simulate_1day`. No registry or runtime protocol check is involved; an incompatible object raises at the point where its missing operation is used.

Communication is capability-based. When `communication_rounds` is nonzero, every supplied policy must implement `act_send(round)` and `act_receive(round, messages)`. The lifecycle does not use an `isinstance` check to decide whether an external policy may communicate.

Both hooks have the signature `hook(results, environment, robots, estimator, evaluator)`. The simulation function returns results but performs no persistence. `save_simulation_results` saves exactly the dictionary selected by its caller.

Visualization code plots score events at their recorded timestamps. It does not backfill scores into earlier timesteps.

## Verification command

Run the complete baseline from the repository root with the project virtual environment:

```shell
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/wbf-matplotlib /Users/lboloni/Documents/Develop/VirtualEnvs/WBF/bin/python -m unittest discover -s unittests -v
```

The suite uses explicit random seeds and temporary directories. It does not use the machine-specific Waterberry Farms configuration or existing experiment data.
