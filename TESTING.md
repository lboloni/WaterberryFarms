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
- Single-robot and multi-robot simulation.
- Perfect communication and MRMR market primitives.
- Result serialization.

Notebooks, files under `obsolete`, LAIP, Bounomodes, and unfinished paper prototypes are not part of the supported test surface.

## Current simulation lifecycle

The current one-day runners use this order at each timestep:

1. Perform configured communication rounds in a multi-robot run.
2. Ask each policy to schedule actions.
3. Execute the robot's pending and every-step actions.
4. Observe the environment at the resulting position.
5. Add the observation to the shared estimator and the robot policy.
6. Update the estimator and score at `im_resolution` intervals and at the final timestep.

The one-day runners treat the environment as static during robot movement. They advance it to `time-start-environment` before the run begins. Changing this lifecycle belongs to modernization Step 2.

## Verification command

Run the complete baseline from the repository root with the project virtual environment:

```shell
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/wbf-matplotlib /Users/lboloni/Documents/Develop/VirtualEnvs/WBF/bin/python -m unittest discover -s unittests -v
```

The suite uses explicit random seeds and temporary directories. It does not use the machine-specific Waterberry Farms configuration or existing experiment data.
