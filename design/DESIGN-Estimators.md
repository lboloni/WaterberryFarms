# DESIGN: Estimators in the Waterberry Farms simulator

This document describes the estimators (information models) as they currently exist in the code, and lists the known issues. It does not propose code changes. Related: [DESIGN-VoI.md](DESIGN-VoI.md), which consumes the estimators' `value` and `uncertainty`.

## 1. Class hierarchy

All low-level estimators are in `information_model.py`; the WBF-specific wrappers are in `water_berry_farm.py`.

```
InformationModel                       add_observation(obs), proceed(delta_t) — both no-ops
└── StoredObservationIM                appends every observation to self.observations
    ├── AbstractScalarFieldIM          self.value, self.uncertainty arrays (width x height)
    │   ├── GaussianProcessScalarFieldIM
    │   ├── PointEstimateScalarFieldIM
    │   └── DiskEstimateScalarFieldIM
    └── WaterberryFarmInformationModel holds im_tylcv, im_ccr, im_soil
        ├── WBF_IM_DiskEstimator       name "AD"
        └── WBF_IM_GaussianProcess     name "GP"
```

### Common interface (`AbstractScalarFieldIM`)

- `__init__(width, height, default_value)` sets `value` to `default_value` everywhere and `uncertainty` to 1 everywhere.
- `add_observation(obs)` only stores the observation. An observation is a dict with keys `x`, `y`, `value` and `time`. Constants such as `CONFIDENCE` and `RANGE` are declared but not used by any estimator.
- `proceed(delta_t)` recomputes from scratch: `self.value, self.uncertainty = self.estimate(self.observations, None, None)`. `delta_t` is ignored and all observations are reprocessed every time.
- `estimate(observations, prior_value, prior_uncertainty) -> (value, uncertainty)` is implemented by each subclass.
- `estimate_voi(observation)` returns the total absolute reduction of `uncertainty` from adding one observation. Nothing calls it.

The base class docstring leaves the meaning of `uncertainty` open ("FIXME: what exactly the uncertainty measures???"), and each subclass gives it a different meaning (Section 2).

## 2. The three scalar-field estimators

### 2.1 `PointEstimateScalarFieldIM`

- **Value:** the observed value at each observed cell, otherwise the prior or `default_value`. A later observation of the same cell overwrites an earlier one.
- **Uncertainty:** 0 at observed cells, otherwise the prior or 1.
- **Priors:** supported (copies `prior_value` / `prior_uncertainty`).
- **Use:** no WBF wrapper uses it. It is useful as a reference and as the "observed" mask.

### 2.2 `DiskEstimateScalarFieldIM` (Adaptive Disk, "AD")

- **Value:** each observation paints its value onto a disk of radius `r` around `(int(x), int(y))`. Later disks overwrite earlier ones, and cells outside every disk keep `default_value`.
- **Uncertainty:** 0 inside any disk, 1 outside.
- **Radius:** with `disk_radius=None` (what `WBF_IM_DiskEstimator` passes) the radius is adaptive, `r = int(1 + sqrt(2·W·H / (π·n)))` with `n = len(observations)`. The total disk area is then about twice the field. Because `n` counts duplicate observations, a robot that stays in place shrinks the radius.
- **Priors:** the arguments are accepted and ignored.
- **Cost:** O(n·r²) per `proceed`, which is cheap.

### 2.3 `GaussianProcessScalarFieldIM` ("GP")

- **Model:** `sklearn.GaussianProcessRegressor` with kernel `RBF(length_scale=[2,2], bounds [1,10]) + WhiteKernel(noise_level=0.5)` (or a user-supplied `gp_kernel`), `n_restarts_optimizer=5`, `random_state=0`, and `normalize_y=False` (the default).
- **Fitting:** observation coordinates are rounded and the hyperparameters are re-optimized on every `proceed`. The GP is then queried at every cell of the grid.
- **Value:** the posterior mean. With no observations it is `default_value`.
- **Uncertainty:** the posterior standard deviation. With no observations it is 1.
- **Priors:** raises an exception if a prior is given.
- **Cost:** O(n³) fit times 6 optimizer starts, plus a prediction on all W·H cells, on every `proceed`. `n` includes duplicates and grows over the whole run.

## 3. WBF wrappers

`WaterberryFarmInformationModel` sends each part of a WBF observation (`"TYLCV"`, `"CCR"`, `"Soil"`) to the matching sub-model, and `proceed` calls all three. Defaults:

| Wrapper | Sub-model | `default_value` for TYLCV / CCR / Soil |
|---|---|---|
| `WBF_IM_DiskEstimator` | `DiskEstimateScalarFieldIM(disk_radius=None)` | 1.0 / 1.0 / 0.0 |
| `WBF_IM_GaussianProcess` | `GaussianProcessScalarFieldIM` | 1.0 / 1.0 / 0.0 |

For the disease fields, 1.0 means healthy, 0.5 infected and 0.0 destroyed (`EpidemicSpreadEnvironment.create_value`, `environment.py`). The defaults therefore assume "healthy until observed otherwise". `WaterberryFarmInformationModel.visualize` is an empty stub.

## 4. How estimators are used

- **Selection:** since Section 7 was implemented, every exprun-driven entry point builds its estimator with `wbf_helper.create_estimator(exp_estimator, geometry)`. Before that, the YAMLs contained only `estimator-name`; `notebooks/1Robot1Day-Run.ipynb` chose the class with an if/else on `run_estimator`, and the other entry points always built `WBF_IM_DiskEstimator`.
- **Simulation loop:** `wbf_simulate.py` calls `estimator.add_observation` for every robot's observation at every timestep. Every `estimator_interval` steps and on the last step it calls `estimator.proceed(interval_count)` and then `evaluator.score(environment, estimator)`.
- **Scores:** the `WBF_Score_*` classes in `water_berry_farm.py` compare `im.value` with `env.value` (L1, weighted, asymmetric). None of them uses `uncertainty`.
- **Policies:** the confidence-guided policy (`papers/y2023_confidenceguided/confidence_guided_ipp_policy.py`) embeds its own `WBF_IM_GaussianProcess` and moves to the feasible waypoint with the highest `im_tylcv.uncertainty`.
- **Figures:** `wbf_figures.graph_env_im` plots `value` and `uncertainty` with `vmin=0, vmax=1`.

## 5. Known issues

1. **Each estimator's uncertainty means something different.** Point and disk uncertainty is a 0/1 coverage indicator. GP uncertainty is a posterior standard deviation in value units. Scores, policies and figures treat them as interchangeable, but the numbers can't be compared across estimators.
2. **GP prior mean is 0, not `default_value`.** Because of `normalize_y=False`, the GP mean decays towards 0 away from the observations. For the disease fields 0 means "destroyed", so the GP predicts disease in unexplored areas. `default_value` only applies when there are no observations at all.
3. **GP standard deviation is not in [0,1].** Far from the data it is about sqrt(1 + white noise), which is above 1 at the initial noise level of 0.5. The `WhiteKernel` noise is included in the predicted std, so it never reaches 0 even at observed cells. The figures clip it at 1.
4. **The GP gets slower over the run.** Duplicate observations are kept, and the GP refits from scratch with 5 restarts each time, so the cost grows as O(n³).
5. **Disk estimator is overconfident.** Every cell inside a disk gets uncertainty 0 regardless of distance, and with the adaptive radius disks cover about twice the field. The last-written disk wins, so conflicting observations never raise uncertainty.
6. **Estimators assume a static field.** No estimator uses `time` or ages its observations, and `proceed`'s `delta_t` is ignored. This is consistent with the current one-day, static-environment runs.
7. **Values are regressed, not classified.** The disease fields take three values (1.0 / 0.5 / 0.0), but every estimator returns a continuous value with no probability of infection. Anything that needs p(infected), such as the VoI design, has to derive it.
8. ~~**Selection is inconsistent.**~~ Resolved by Section 7: some entry points used to ignore `run_estimator`, so an exprun could name the GP and still run the disk estimator.
9. **`estimate_voi` is unused** and, for the GP, costs two full refits per call.

## 6. Possible directions (not proposed for implementation here)

- Give `uncertainty` one documented meaning, e.g. a confidence in [0,1] or a calibrated std, and normalize the GP std to it.
- Use `normalize_y=True`, or fit the GP on `value - default_value`, so that the prior mean is `default_value`.
- De-duplicate observations per cell before fitting.
- Make the estimators configurable from expruns through a single factory (designed in Section 7).

## 7. Design: configuring estimators from expruns

**Status:** implemented. This deliberately reverses, for estimators only, the rule from commit 2d67718 that configuration files do not select Python implementations. `README.md` ("Explicit simulation API") records the exception, and `unittests/test_explicit_construction.py` still guards against a `create_policy` or `create_score` factory.

**Goal:** the estimator exprun fully determines which estimator runs and how. With the YAML files below, every existing exprun reproduces today's behavior exactly; issue 8 (inconsistent selection) is fixed; and changes such as `normalize_y` (issue 2) or a fixed disk radius become config changes.

### 7.1 Estimator YAMLs

The estimator type becomes an explicit field rather than being inferred from the run name. Variants such as `adaptive-disk-r5` can then be added without code changes. Policies use the run-name approach (`startswith` on `run-policy` in `run_experiment.py`), but an explicit type field is more robust.

`data/expruns/estimator/_defaults_estimator.yaml` holds the fields shared by all estimators. `Config.get_experiment` merges it under the run file (`group_config | indep_config`), so every exp gets these fields:

```yaml
input-to-notebook: []
default-tylcv: 1.0   # healthy
default-ccr: 1.0     # healthy
default-soil: 0.0
```

`adaptive-disk.yaml`:

```yaml
estimator-name: "AD"
estimator-type: "disk"
disk-radius: null    # null = adaptive radius
```

`gaussian-process.yaml`:

```yaml
estimator-name: "GP"
estimator-type: "gaussian-process"
gp-length-scale: 2.0
gp-length-scale-bounds: [1, 10]
gp-noise: 0.5
gp-restarts: 5
gp-normalize-y: false
```

Example of a new variant, which needs no code change, `disk-r5.yaml`:

```yaml
estimator-name: "D5"
estimator-type: "disk"
disk-radius: 5
```

### 7.2 Constructor changes (`water_berry_farm.py`, `information_model.py`)

The wrappers take the parameters they currently hard-code, with the current values as defaults, so code that builds them directly keeps working:

```python
class WBF_IM_DiskEstimator(WaterberryFarmInformationModel):
    def __init__(self, width, height, disk_radius = None,
                 default_tylcv = 1.0, default_ccr = 1.0, default_soil = 0.0):
        ...

class WBF_IM_GaussianProcess(WaterberryFarmInformationModel):
    def __init__(self, width, height, gp_kernel = None, gp_restarts = 5, gp_normalize_y = False,
                 default_tylcv = 1.0, default_ccr = 1.0, default_soil = 0.0):
        ...
```

`GaussianProcessScalarFieldIM.__init__` gains `n_restarts_optimizer = 5` and `normalize_y = False`, which are passed through to `GaussianProcessRegressor`. The kernel is already a parameter (`gp_kernel`).

### 7.3 Factory (`wbf_helper.py`)

`wbf_helper.py` already holds the exprun-driven factories (`create_wbf`, `create_wbfe`), so the new one goes next to them:

```python
def create_estimator(exp_estimator, geometry):
    """Factory function for creating a WBF estimator from an estimator exp"""
    if exp_estimator["estimator-type"] == "disk":
        estimator = WBF_IM_DiskEstimator(
            geometry["width"], geometry["height"],
            disk_radius=exp_estimator["disk-radius"],
            default_tylcv=exp_estimator["default-tylcv"],
            default_ccr=exp_estimator["default-ccr"],
            default_soil=exp_estimator["default-soil"])
    elif exp_estimator["estimator-type"] == "gaussian-process":
        kernel = RBF(length_scale=[exp_estimator["gp-length-scale"]] * 2,
                     length_scale_bounds=exp_estimator["gp-length-scale-bounds"]) \
            + WhiteKernel(noise_level=exp_estimator["gp-noise"])
        estimator = WBF_IM_GaussianProcess(
            geometry["width"], geometry["height"], gp_kernel=kernel,
            gp_restarts=exp_estimator["gp-restarts"],
            gp_normalize_y=exp_estimator["gp-normalize-y"],
            default_tylcv=exp_estimator["default-tylcv"],
            default_ccr=exp_estimator["default-ccr"],
            default_soil=exp_estimator["default-soil"])
    else:
        raise Exception(f"Unknown estimator type {exp_estimator['estimator-type']}")
    estimator.name = exp_estimator["estimator-name"]
    return estimator
```

As AGENTS.md requires, there is no `.get()` with fallbacks: a missing field raises a `KeyError`.

### 7.4 Call sites

Every exprun-driven entry point replaces its hard-coded construction (and its `estimator.name = ...` line) with `estimator = create_estimator(exp_estimator, geometry)`:

- `papers/y2025_mrmr/run_experiment.py` and `papers/y2027_mrmr/run_experiment.py`. These currently always use the disk estimator.
- `notebooks/1Robot1Day-Run.ipynb`, which replaces its if/else on `run_estimator`.
- `notebooks/1Robot1Day-Subruns.ipynb`, `notebooks/nRobot1Day-Run.ipynb` and `notebooks/CustomWBFE-experiments.ipynb`. These load `exp_estimator` but build the estimator by hand.

Code that builds estimators directly without an estimator exprun stays unchanged: the confidence-guided paper, `y2023_glr`, `unittests/test_simulation.py`, `examples/external_components.py` and the older experiment notebooks.

### 7.5 Behavior change to note

The MRMR runs currently ignore `run_estimator` and always use the disk estimator. After this change, an MRMR exprun that names `gaussian-process` will really run the GP. Currently every exprun under `data/expruns/mrmr*` names `adaptive-disk`, so no existing MRMR result changes.

### 7.6 Verification

1. `python -m unittest discover unittests` still passes. `test_estimator_factory` covers the factory.
2. Run one `adaptive-disk` exprun before and after the change. The score events must be identical.
3. Run `notebooks/1Robot1Day-Run.ipynb` with `run_estimator: "gaussian-process"` before and after. The score events must be identical, since `gp-normalize-y: false` keeps the current behavior.
4. Run `disk-r5` and check that `im_tylcv.mask_radius == 5`.
