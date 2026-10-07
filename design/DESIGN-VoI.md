# DESIGN: Value of Information (VoI) scores

Source of the ideas: `Lotzi-WaterberryFarms-Settings/TODO.md`, section "New ideas for code / VoI".

## 1. Definitions

For a disease field (TYLCV on tomato, CCR on strawberry) every cell is either infected or not.

- `v_pos` (e.g. +100): value of knowing a cell is infected (cost of damage avoided).
- `v_neg` (e.g. +1): value of knowing a cell is not infected (cost of verification saved).
- `v_unknown` (e.g. -3, conventionally <= 0): cost of ignorance for an unobserved cell.
- `p_pos`, `p_neg = 1 - p_pos`: the estimated probability that a cell is infected.
- `c` in [0,1]: the estimator's confidence at a cell.

The four VoI variants:

| Variant | Value of a cell | Sum over the crop mask |
|---|---|---|
| Absolute | observed: `v_pos` or `v_neg` (ground truth); unobserved: 0 | `voi-absolute` |
| Expected | observed: as absolute; unobserved: `p_pos v_pos + p_neg v_neg` | `voi-expected` |
| Cost of ignorance | observed: as absolute; unobserved: `v_unknown` | `voi-ignorance` |
| Estimator-based | `c (p_pos v_pos + p_neg v_neg) + (1 - c) v_unknown` | `voi-estimator` |

Estimator-based VoI contains the others as special cases:
- `c = 1` gives expected VoI.
- `c = 0` gives cost of ignorance.
- An observed cell (`c = 1`, `p_pos` equal to 0 or 1) gives absolute VoI.

**Incremental VoI** is the difference between two successive values of the same variant. The incremental absolute VoI of an observation is `v_pos` or `v_neg` on the first visit and 0 on a repeat visit. The incremental estimator-based VoI can be negative, for example when an observation makes the estimator less certain about other cells.

### Relation to the literature (short)

- Absolute VoI is a prize-collecting / orienteering objective (a reward is collected once, on the first visit). With `v_neg = 0` it is the objective of *active search* (Garnett et al., ICML 2012).
- Expected VoI is the expected reward used in Bayesian search theory and in informative path planning with a spatial prior.
- In decision-theoretic VoI (Howard 1966; Krause & Guestrin, UAI 2005 / JAIR 2009), the cost of ignorance is not a constant. It is the Bayes risk of acting optimally under the current belief, about `-min(p_pos v_pos, p_neg v_neg)`. **Optional variant:** replace the constant `v_unknown` with this per-cell Bayes risk (`v_unknown = "bayes-risk"`). This makes the cost of ignorance depend on the estimator, so it rewards more than just not revisiting cells.
- The `c`-weighted mixture is a heuristic. Unlike the expected value of sample information, it is not guaranteed to be non-negative in expectation.

## 2. Implementation: one new score class

Add `WBF_Score_VoI(WBF_Score)` to `water_berry_farm.py`, after `WBF_MultiScore` (line 407). It follows the style of `WBF_Score_WeightedAsymmetric` (keyword arguments with commented defaults, plus `__str__`) and of `WBF_MultiScore` (`score` returns a dict, and a static `score_components()` lists the keys).

```python
class WBF_Score_VoI(WBF_Score):
    """Value of information scores (absolute, expected, ignorance, estimator-based) for the disease fields. See design/DESIGN-VoI.md"""
    def __init__(self, v_pos = 100.0, # knowing it is infected: cost of damage
                 v_neg = 1.0, # knowing it is not infected: cost of verification
                 v_unknown = -3.0): # cost of ignorance
        ...

    @staticmethod
    def score_components():
        return ["voi-absolute", "voi-expected", "voi-ignorance", "voi-estimator", "voi"]

    def score(self, env, im):
        retval = {}
        for name in self.score_components():
            retval[name] = 0.0
        for envfield, imfield, mask in [(env.tylcv, im.im_tylcv, env.my_tomato_mask),
                                        (env.ccr, im.im_ccr, env.my_strawberry_mask)]:
            for name, value in self.field_voi(envfield, imfield, mask).items():
                retval[name] += value
        retval["voi"] = retval["voi-estimator"]  # the headline component
        return retval
```

`field_voi(envfield, imfield, mask)` works on numpy arrays of shape `(width, height)`:

- `infected = envfield.value < 1.0`. Ground truth: 1.0 is healthy, 0.5 infected, 0.0 destroyed (`EpidemicSpreadEnvironment.create_value`, `environment.py`). Destroyed counts as infected.
- `observed`: a boolean array set to True at each `(o["x"], o["y"])` in `imfield.observations` (`StoredObservationIM`, `information_model.py:50`).
- `p_pos = np.clip(1.0 - imfield.value, 0, 1)`. The disease sub-models default to 1.0 (healthy), so an estimate of 1.0 means `p_pos = 0` and an estimate of 0.5 or lower means `p_pos` of at least 0.5.
- `c = np.clip(1.0 - imfield.uncertainty, 0, 1)`. This is exact for the disk and point estimators, whose uncertainty is in {0, 1}. For the GP, the standard deviation is used directly (see the open questions).
- `truth = np.where(infected, v_pos, v_neg)`
- `expect = p_pos * v_pos + (1 - p_pos) * v_neg`
- The four components are then sums restricted to `mask`:
  - absolute: `sum(mask * observed * truth)`
  - expected: `absolute + sum(mask * ~observed * expect)`
  - ignorance: `absolute + v_unknown * sum(mask * ~observed)`
  - estimator: `sum(mask * (c * expect + (1 - c) * v_unknown))`

Soil is not included, because the infected / not-infected model does not apply to it.

### Incremental VoI

No new code is needed. `wbf_simulate.py` (around line 137) already calls `evaluator.score(environment, estimator)` every `estimator_interval` steps and on the last step, and appends the result to `results["score-events"]`. Incremental VoI is the difference between consecutive events. Taking this difference in the analysis / figures code keeps the simulation loop unchanged.

## 3. Configuration

New file `data/expruns/score/voi.yaml`:

```yaml
score-name: "voi"
v-pos: 100.0
v-neg: 1.0
v-unknown: -3.0
```

In `papers/y2027_mrmr/run_experiment.py` (lines 33-34), the score class is currently hard-coded to `WBF_Score_WeightedAsymmetric()`. Choose the class by `exp["run_score"]` instead:

```python
if exp["run_score"] == "voi":
    evaluator = WBF_Score_VoI(exp_score["v-pos"], exp_score["v-neg"], exp_score["v-unknown"])
else:
    evaluator = WBF_Score_WeightedAsymmetric()
evaluator.name = exp_score["score-name"]
```

Because `score` returns a dict, code that reads `results["score"]` as a float must use `results["score"]["voi"]`, as it already does for `WBF_MultiScore`.

## 4. Changes summary

| File | Change |
|---|---|
| `water_berry_farm.py` | + `WBF_Score_VoI` (about 40 lines) |
| `data/expruns/score/voi.yaml` | new |
| `papers/y2027_mrmr/run_experiment.py` | if/else on `run_score` |

## 5. Open questions

1. **Confidence `c`:** is it per cell (as proposed) or one global value per estimator? A per-cell `c` is what lets a confusing observation lower the VoI elsewhere.
2. **GP uncertainty:** the GP standard deviation is not bounded to [0,1]. Should it be normalized, for example by the prior standard deviation?
3. **Destroyed cells:** are destroyed cells (value 0.0) worth `v_pos`, or something else, since there is nothing left to save?
4. **Crop weights:** should tomato and strawberry have separate `v_pos` / `v_neg`, like the importances in `WBF_Score_WeightedAsymmetric`?
5. **Cost of ignorance:** should the default be the constant `v_unknown` or the Bayes-risk variant?
