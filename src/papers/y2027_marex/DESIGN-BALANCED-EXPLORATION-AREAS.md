# Design: balanced exploration areas for multi-robot baselines

## Purpose

The MAREX comparisons need stronger baselines (TODO-Submission-ICC2027.md, "New baselines"): a balanced multi-robot lawnmower, and balanced versions of the other explorers, such as random waypoint and confidence-guided exploration. Each of them divides the field into one area per robot, and every robot explores its own area with some technique.

A robot first has to travel to its area. Its exploration therefore has two parts: the travel to the area, then the exploration of the area. Robots whose areas are farther away spend more of the same budget on travel. They should get smaller areas, so that every robot explores its area with the same density.

This design defines that allocation and the policies that use it:

- The areas are computed from a **lawnmower equivalent**: the areas for which every robot, after its travel, could cover its area with a lawnmower of the same pass spacing.
- The algorithm that actually explores an area can be different (lawnmower, random waypoint, confidence-guided). The allocation only fixes the area and the budget left for it.
- When the budget runs out, the robot can be anywhere. It does not have to return.

## 1. The current multi-robot lawnmower (MRSE)

- **Setup:**
  - The MRSE runs (`marex2027-run/*-mrse`) divide the field into equal horizontal strips, one per robot, separated by gaps (e.g. `[10, 10, 190, 60]`, `[10, 70, 190, 120]`, `[10, 130, 190, 190]` on 200x200).
  - All robots start at the home base `(0, 0)` (`Robot(name, 0, 0, 0)` in `run_experiment.py`).
  - Every robot has a budget of 1000 timesteps at velocity 1.
- **Path:** `generate_fixed_budget_lawnmower` (`wbf_helper.py`) calls `find_fixed_budget_lawnmower` (`path_generators.py`). That function searches for the largest number of **winds** (two passes each) whose path, *including the travel from the start*, fits into the budget.
- **What it already does:** it charges the travel against the budget.
- **What it does not do:** it does not adapt the areas to the travel.

Measured on the current configurations (pass spacing = strip height / (2 × winds); budget used = travel + path):

| Field, robots | Strip (y) | Travel | Winds | Pass spacing | Budget used |
|---|---|---|---|---|---|
| 100x100, 3 | 5–30 | 7 | 4 | 3.1 | 842 |
| | 35–60 | 35 | 4 | 3.1 | 870 |
| | 65–95 | 65 | 4 | 3.8 | 905 |
| 200x200, 3 | 10–60 | 14 | 2 | 12.5 | 964 |
| | 70–120 | 71 | 1 | **25.0** | **661** |
| | 130–190 | 130 | 1 | **30.0** | 730 |
| 200x200, 6 | 10–35, 41–66, 72–97 | 14–73 | 2 | 6.2 | 939–998 |
| | 103–128, 134–159, 165–190 | 103–165 | 1 | **12.5** | **668–730** |

Three problems:

1. **The density is uneven.** On 200x200 the far strips are covered with a 2–2.4× coarser pass spacing than the near ones.
2. **Budget is wasted.** The number of passes can only grow in whole winds (two passes). A robot whose budget is not enough for another wind leaves up to a third of its budget unused (661 of 1000).
3. **The unused budget re-covers the strip.** The path is followed with `FollowPathPolicy(..., repeat=True)`, so after the last pass the robot flies back to the start of its strip and covers it again.

## 2. The model

- **Field:** the area to explore is a rectangle `[x0, x1] × [y0, y1]`, with width `W = x1 − x0` and height `Y = y1 − y0`.
- **Robots:** n robots with the same velocity `v` and the same time budget `T`, i.e. the same distance budget `D = v·T`. Robot i starts at `start_i`; currently all start at `(0, 0)`.
- **Lawnmower equivalent:** a strip is covered by horizontal **passes** of length `W`, `s` apart (the pass spacing), connected at alternating ends. Covering p passes after a travel of `d` costs

  `cost(d, p, s) = d + p·W + (p − 1)·s`

  The travel `d` is the distance from the start to the **entry point**: the end of the strip's first pass that is nearer to the start.
- **Balance criterion:** every robot covers its strip with the **same pass spacing** `s`. Equal spacing means equal lawnmower density, i.e. every part of the field gets the same coverage per unit area, whichever robot explores it. The team spacing `s` should be as small as possible, given that the strips together cover the whole field.
- **End of the budget:** the robot can stop anywhere. There is no return leg in the cost.

## 3. The allocation

### 3.1 Continuous version (for intuition)

If the number of passes could be fractional, a strip of height `H_i` with spacing `s` has `H_i/s + 1` passes. The budget constraint gives

`H_i = (D − d_i − W) / (W/s + 1)`

The robots are stacked away from the start (strip 1 nearest). Adjacent strips are `s` apart, so that the passes are evenly spaced across strip borders. Strip i then starts at `y0 + Σ_{j<i} (H_j + s)`. That position determines `d_i`, and `d_i` determines `H_i`. The total height grows with `s`, so a bisection on `s` finds the spacing at which the strips exactly fill `[y0, y1]`. A farther robot has a larger `d_i` and therefore a smaller `H_i`.

### 3.2 Integer passes (what is implemented)

A real lawnmower has a whole number of passes. Rounding each strip separately would make the spacing uneven again. Instead, the allocation lays **one global lawnmower** over the field and divides its passes among the robots:

- The global lawnmower has `P` passes at `y0 + k·s`, `k = 0 … P−1`, with `s = Y/(P − 1)`.
- Robot i gets a **block of consecutive passes** `k_i … k_i + p_i − 1`. Its entry point is the nearer end of pass `k_i`.
- **Greedy assignment for a given s**, from the nearest block to the farthest:

  `p_i(s) = floor((D − d_i + s) / (W + s))`, the most passes that satisfy `cost(d_i, p_i, s) ≤ D`, where `d_i` is the travel to pass `k_i = Σ_{j<i} p_j`.

- **The team spacing:** `P` is the largest number of passes such that the greedy assignment with `s = Y/(P − 1)` provides at least `P` passes. The last blocks are trimmed so that exactly `P` passes are assigned.
- **Infeasible robots:** if a robot cannot afford a single pass (`p_i = 0`), it gets no area. The allocation reports this, because it means the field is too large for the team and the budget.

**Properties:**
- The spacing is exactly uniform over the whole field.
- Farther robots get fewer passes.
- Every robot except the trimmed ones leaves less than one pass (`W + s`) of its budget unused.

**Applied to the current configurations** (area as in the MRSE runs, budget 1000):

| Field, robots | Passes, spacing | Blocks (passes per robot, from near to far) | Travel | Budget used |
|---|---|---|---|---|
| 100x100, 3 | 30, 3.10 | 10, 10, 10 | 7, 36, 67 | 935, 964, 995 |
| 200x200, 3 | 13, **15.0** | 5, 4, 4 | 14, 86, 145 | 974, 851, 910 |
| 200x200, 6 | 26, **7.2** | 5, 5, 4, 4, 4, 4 | 14–169 | 824–976 |

Compared with the equal strips, the spacing is uniform:
- 200x200, 3 robots: 15.0 everywhere, instead of 12.5 / 25 / 30;
- 200x200, 6 robots: 7.2 everywhere, instead of 6.2 / 12.5.

On 100x100 the equal strips were already nearly balanced, because the travel is small compared with the budget.

### 3.3 The areas

The **area** of robot i is the strip its passes cover: `[x0, x1] × [y(k_i) − s/2, y(k_i + p_i − 1) + s/2]`, clipped to the field. Adjacent areas touch, so together they cover the field without overlap. The allocation returns, for every robot:

| Key | Value |
|---|---|
| `area` | `[xmin, ymin, xmax, ymax]` of its strip |
| `entry` | the entry point, the nearer end of its first pass |
| `travel` | `d_i`, the distance from its start to the entry point |
| `budget` | `D − d_i`, the budget left for exploring the area |
| `passes` | its block of passes `(k_i, p_i)` and the team spacing `s` |

## 4. Exploring the areas

The allocation fixes *where* a robot explores and *how much* budget it has there. The explorer decides *how*:

| Policy (`run-policy`) | Explorer in the area |
|---|---|
| `balanced-lawnmower` | follows its block of passes: the balanced multi-robot lawnmower |
| `balanced-random-waypoint` | `RandomWaypointPolicy` confined to its area |
| `balanced-confidence-guided` | `ConfidenceGuidedPathPlanning` (`papers/y2023_confidenceguided/`) with `low_point`/`high_point` set to its area |

- **The travel phase.** A new wrapper, `TransitThenExplorePolicy(entry, explorer)`, flies straight to the entry point, then hands over to the explorer.
  - The lawnmower and random waypoint would reach their areas by themselves, but the confidence-guided policy only chooses waypoints within a small span around the robot, so it would never leave the home base on its own.
  - The wrapper makes the travel explicit and the same for every explorer. It also lets the figures mark where the travel ends.
- **The balanced lawnmower** uses `FollowPathPolicy(..., repeat=False)`. When its passes are done (less than one pass of budget left), it hovers in place. It does not re-cover its strip.
- **The confidence-guided explorer** (`AreaConfidenceGuided`) chooses, within a span of 5 cells around the robot and inside its area, the cell of the highest uncertainty of its own estimator. Its waypoint search is robust to non-integer positions and to a robot outside its area (it heads to the nearest cell of the area).
  - With span 5, it moves in short steps, lingers near its entry point, and covers only part of its strip: on a clustered 200x200 map it observed about 2,100 cells, against about 2,700 for the other explorers.
  - A larger span is the knob to try if it is meant to be a strong baseline.

## 5. Configuration

The allocation is a property of the team, not of one robot, so it is declared at the run level:

```yaml
# e.g. marex2027-run/clustered-200-bmrse.yaml
approach: "bmrse"
balanced-areas:
  area: [10, 10, 190, 190]        # the rectangle the team divides; default: the field
  robots: [lawn-1, lawn-2, lawn-3] # the order of the blocks, nearest first; default: all robots, in order
robots:
  - name: "lawn-1"
    exp-policy: "policy"
    run-policy: "balanced-lawnmower"
    exp-policy-extra-parameters:
      policy-name: "balanced-lawn-1"
      budget: 1000
  ...
```

- `run_experiment.py` computes the allocation once per run, from the robots' budget, the geometry's velocity and the robots' starts. It passes each robot its entry of the allocation (`exp_policy["allocation"]`) when constructing its policy.
- The allocation is saved in the results (`results["balanced-areas"]`), so the figures can draw the areas.

## 6. Code changes (implemented)

| File | Change |
|---|---|
| `src/papers/y2027_marex/balanced_areas.py` | `allocate_balanced_strips(starts, area, velocity, budget)` (Section 3.2); `lawnmower_passes(allocation, area)`, the waypoints of a robot's block of passes; `TransitThenExplorePolicy(vel, entry, explorer)`; `AreaConfidenceGuided`; `create_balanced_policy(run_policy, exp_policy, allocation, area, velocity, estimator)` |
| `src/papers/y2027_marex/run_experiment.py` | `balanced_allocation(exp, geometry)`: the team-level allocation of a run with `balanced-areas`; constructing the three `balanced-*` policies; `results["balanced-areas"]` |
| `src/papers/y2027_marex/marex_metrics.py` | the roles `balanced-lawnmower`, `balanced-random-waypoint`, `balanced-confidence-guided` |
| `data/expruns/policy/` | `balanced-lawnmower.yaml`, `balanced-random-waypoint.yaml` (seed), `balanced-confidence-guided.yaml` (seed, span 5) |
| `data/expruns/marex2027-run/` | the 18 balanced base runs `{clustered, unclustered}{,-200,-200-6}-{bmrse, bmrrw, bmrcg}` |
| `src/test/test_balanced_areas.py` | the tests of Section 7 |

**Implementation notes:**
- **Robot assignment:** `Robot.assign_policy` sets `policy.robot` directly, without calling `assign()`. `TransitThenExplorePolicy.robot` is therefore a property that also sets the robot of the explorer.
- **Estimator of the confidence-guided explorer:** each robot gets a fresh estimator of the run's type (`run_estimator`, `adaptive-disk`), not the three-field Gaussian process of the 2023 policy, which would be far slower. The choice function is plain uncertainty (`choose_next_waypoint` of the 2023 policy, without its print). The `x > 15` weighting was only in the unused `choose_next_waypoint_exp`.
- **Approach keys:**
  - `bmrse`: balanced multi-robot lawnmower;
  - `bmrrw`: balanced random waypoint;
  - `bmrcg`: balanced confidence-guided.
- **In the paper flow (2026-10-12):** the old MRSE (equal strips, `fixed-budget-lawnmower`) is removed from the 2027 definitions. The baselines are now MRRW, BMRSE (`bmrse`) and BMRCG (`bmrcg`), in the aggregate `icc-2027-replicated`, both flows, the figures and the sync script. The balanced random waypoint runs (`bmrrw`) exist, but are not in the flows.

**Cost:** 3 balanced approaches × 2 maps × 3 settings (100x100 1+2, 200x200 1+2, 200x200 1+5) give 18 new base runs. At 20 replications that is 360 more runs, which doubles the replicated part of the paper flow. A balanced confidence-guided run takes about 2 s with 3 robots and 5 s with 6; the others take about 1–2 s.

## 7. Tests

- **The allocation** (`test/test_balanced_areas.py`):
  - every robot's cost is within its budget;
  - the passes are contiguous blocks covering `y0 … y1` with one uniform spacing;
  - the passes per robot do not increase with the travel;
  - the areas touch and cover the field;
  - for the three configurations above, the allocation reproduces the tables of Section 3.2;
  - a robot that cannot afford a pass gets none, and is reported.
- **The path:** the length of `lawnmower_passes` plus the travel equals `cost(d, p, s)`.
- **The policies:** a short simulation per `balanced-*` policy. Every robot reaches its entry point at timestep `travel / v` and then stays inside its area.
- **A figure:** the allocated strips and the trajectories, in the style of `detection-map-*`, as a check that the areas look right.

## 8. Open questions

1. ~~**Replace or add?**~~ Decided: the balanced lawnmower replaces MRSE.
2. **Different starts:** with robots starting at different points, the assignment of blocks to robots becomes an assignment problem (e.g. minimal total travel). Today all robots start at the home base, so the order is the distance order.
3. **The shape of the areas:** horizontal strips stacked away from a corner start are the simplest. From a corner, the travel grows with both x and y. Dividing into rows *and* columns, or fan-shaped areas, would reduce the travel to the far areas. That is worth considering only if the strips leave too much budget in travel.
4. **The leftover budget:** less than one pass per robot is left. Hovering is the simplest honest choice. Alternatives are a partial last pass, or random waypoints in the area.
5. **Spacing and sensing:** a robot observes one cell per timestep, so any spacing above 1 leaves cells between the passes unobserved. The balance is about *equal* coverage, not complete coverage. If the spacing ever drops below the sensing footprint, cap it at the footprint.
