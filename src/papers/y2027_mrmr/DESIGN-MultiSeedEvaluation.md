# Multi-seed evaluation of the MRMR comparisons

**Status:** implemented on top of the ExpRunFlow library (0.2.0, `ExpRunFlow/docs/DESIGN-Replication.md`), which now contains the general mechanism:
- the generation of the replications (`exprunflow.replication`);
- `metrics.json` (`exprunflow.metrics`);
- the tidy tables and statistics (`exprunflow.aggregate`);
- the standard figures (`exprunflow.plots`).

The project supplies only what is specific to MRMR:
- **The appliers** `apply_map_seed` and `apply_behavior_seed` in `src/wbf_flow.py`: what a map seed and a behavior seed change.
- **The measurements of a run** (`mrmr_metrics.collect_metrics`) and its labels (`map-size`, `scenario`, `approach`).
- **The MRMR-specific figures:** the communication cost, and the VoI per role.

The names in this document have changed in the library. The replication block replaces the top-level `base-run` and `behavior-seed` keys, the `factors` section of the aggregate replaces the `map-seeds` and `behavior-seeds` lists, and the tables use the column names `map-seed`, `entities.csv` with `group`, and `n_clusters`.

**The current experiment** (`mrmr2027-aggregate/icc-2027-replicated`): 12 base runs (100x100 and 200x200 fields, clustered and unclustered, MRMR, MRSE and MRRW) x 5 map seeds x 4 behavior seeds = 240 replications.

## Purpose

The MRMR 2027 comparisons currently run every approach once: one generated map per scenario (seed 1), and one fixed seed per robot. A single run cannot tell whether a difference between MRMR, MRSE and MRRW is a property of the approaches or an accident of one map and one set of random choices. The reviewers asked for multiple runs, and the paper's criticism lists "Run multiple experiments over them with different randomness" and "Create graphs with confidence intervals" (`CRITICISM-Lotzi.md`).

This design runs every comparison as a set of **independent replications**: the same configuration, varying only the random seeds. It then reports means with confidence intervals and ranges. Specifically, it:
- separates the two sources of randomness, the **map** and the **robot behavior**, and gives each its own seed;
- uses **common random numbers**: within a replication, all approaches face the same map and draw from the same behavior seeds, so they can be compared pairwise;
- generates the replications inside the flow, following the "generated expruns" pattern of ExpRunFlow (`ExpRunFlow/docs/DESIGN-Flows.md`). ExpRunFlow itself has no seed-sweep functionality;
- lays the results out as **tidy tables** (one observation per row), from which any graph with confidence intervals or ranges can later be drawn without loading the simulation pickles.

### Terminology

| Term | Meaning |
|---|---|
| scenario | a kind of map: `clustered` or `unclustered` |
| approach | `mrmr`, `mrse` (lawnmowers) or `mrrw` (random waypoints) |
| base run | a checked-in run of `mrmr2027-run`, e.g. `clustered-mrmr` = scenario + approach |
| map seed `m` | the seed of the generated disease map |
| behavior seed `b` | the seed from which the random generators of all robots are derived |
| replication | one run of a base run with one `(m, b)` |
| cell | the set of replications of one scenario and approach; statistics are computed per cell |

## 1. Sources of randomness

| Source | Where | Seed today | Varies with |
|---|---|---|---|
| The disease map | `tylcv-generated-seed` of `environment/mrmr-generated-*-100` | 1 | map seed `m` |
| Other environment fields | `seed` (CCR epidemic) and `soil-seed` of the environment | 10, 1 | constant (not observed by the score on a tomato-only farm) |
| MRMR pioneer and contractors | `exp-policy-extra-parameters.seed` of each robot (`MRMR_Policy.random`: random waypoints, pioneer exploration) | 1, 2, 3 | behavior seed `b` |
| Random waypoint robots (MRRW) | `seed` of each robot (`RandomWaypointPolicy.random`) | 1, 10, 20 | behavior seed `b` |
| Lawnmower robots (MRSE) | none: `fixed-budget-lawnmower` is deterministic | | nothing |
| Estimator (`adaptive-disk`) and VoI score | none | | nothing |
| **EP path search** (`ExplorationPackageSet.find_shortest_path_ep`) | stops after a **wall-clock** `maxtime` (0.1 s in `can_bid`, 1.0 s in `replan`) | none | the **speed and load of the machine** |

**The EP path search is a hidden source of randomness.** It enumerates EP orders and lawnmower directions, and returns the best path found before the time limit. With more than about four committed EPs it does not finish. Its result, and with it the bids, the plans and the trajectories, then depends on how fast the machine is and on what else runs on it. The same seeds would not reproduce the same run, and running replications in parallel would make this worse. Section 8 replaces the time limit with a deterministic limit.

## 2. Seeds

### 2.1 Map seed

A replication with map seed `m` uses an environment exprun whose `tylcv-generated-seed` is `m`. All other keys are those of the scenario's environment. The clustered and unclustered maps of the same `m` are independent maps: `generate_disease_map` derives their generators from `(m, width, height, version)`. They have exactly the same number of diseased plants (DESIGN-ClusteredDiseaseMap.md).

### 2.2 Behavior seed

A replication with behavior seed `b` overrides the seed of every robot whose policy has one. The seed of the robot at position `i` of the run's `robots` list is

```python
robot_seed(b, i) = int(np.random.SeedSequence([b, i]).generate_state(1)[0])
```

- The robots of a team get independent generators, with no accidental shared streams such as seeds 1, 2, 3 and 2, 3, 4 in consecutive replications.
- **Common random numbers:** robot `i` of every approach gets the same seed in replication `b`. Different policies draw differently from it, so this does not make the approaches behave alike. It makes the comparison of an approach with itself across scenarios paired, and it keeps the design simple.
- **Deterministic approaches:** policies without a seed (the lawnmowers) are unaffected, so for MRSE all replications with the same `m` are identical. The design still runs them, as a run takes seconds, so every cell has the same shape. The statistics then correctly show zero behavior variance for MRSE.
- **The base runs stay as they are:** the checked-in runs keep their hand-chosen robot seeds and are not replications. They remain the "single run" of the current figures.

### 2.3 Experimental design

The map seeds and the behavior seeds are **crossed**: every map seed is combined with every behavior seed, so a cell has `|M| x |B|` replications. With both lists in the configuration:
- **`|B| = 1`:** only the map varies.
- **`|M| = 1`:** only the behavior varies, on one map.
- **Both larger than 1:** the variance can be split into its map and its behavior components (Section 6.3).

The default proposal is `M = 1..10`, `B = 1..3`, giving 30 replications per cell. With 2 scenarios and 3 approaches, that is 180 runs. A run takes about 9 seconds on Miniberry-100 (the six runs of the current flow took 56 seconds), so the whole set takes about 30 minutes sequentially.

## 3. Configuration

The replications are declared once, in the **aggregate exp/run**. The flow that generates them and the aggregation that collects them therefore cannot disagree about which runs exist:

```yaml
# data/expruns/mrmr2027-aggregate/icc-2027-replicated.yaml
name: ICC 2027 MRMR, replicated over the robot behavior
runs:                       # the base runs to replicate
  - clustered-mrmr
  - clustered-mrse
  - clustered-mrrw
  - unclustered-mrmr
  - unclustered-mrse
  - unclustered-mrrw
map-seeds: [1]              # the first experiment: the robot behavior only
behavior-seeds: [1, 2, 3]
```

The family defaults (`_defaults_mrmr2027-aggregate.yaml`) set `source-experiment: mrmr2027-run` and the order of the approaches in the paired differences.

The flow exp/run refers to the aggregate:

```yaml
# data/expruns/mrmr2027-flow/icc-2027-replicated.yaml
run-experiment: mrmr2027-run
aggregate-experiment: mrmr2027-aggregate
aggregate: icc-2027-replicated
figure-experiment: mrmr2027-figure
figures: [replicated-comparison-clustered, replicated-comparison-unclustered, replicated-voi-over-time,
          replicated-paired-clustered, replicated-paired-unclustered, replicated-communication-cost,
          replicated-per-role]
```

It runs with `MRMR-Flow.ipynb`, using the parameters `run = "icc-2027-replicated"` and, to keep its workspace apart, `flow_name = "icc-2027-replicated"`.

Explicit seed lists are used rather than a count, so that a set can be extended (adding seeds 11..20) without rerunning the existing replications.

## 4. Generating the replications

`build_mrmr_2027_flow_entries` (`src/wbf_flow.py`) dispatches to `build_mrmr_2027_replicated_entries` when the collection has an `aggregate`. Into the flow workspace (`expruns/`), it writes:

1. **An environment variant per scenario and map seed:** `environment/<env>-m<m>`, e.g. `mrmr-generated-clustered-100-m3`. It is the base environment with `tylcv-generated-seed: m`. One "Precompute" entry is queued per variant.
2. **A run variant per base run, map seed and behavior seed:** `mrmr2027-run/<base>-m<m>-b<b>`, e.g. `clustered-mrmr-m3-b2`. It is the base run with:
   - `run_environment` set to the environment variant;
   - the seed of every robot that has one replaced by `robot_seed(b, i)`. The `robots` list is replaced as a whole, since an exprun variant only changes top-level keys;
   - the identification fields of Section 5.1: `base-run`, `scenario`, `approach`, `map-seed`, `behavior-seed`.

   One "Run" entry is queued per variant.
3. **The aggregation entry** (Section 6), then **the figure entries.**

The variants are written with a small helper, `write_exprun_variant(family, run, changes, new_run)`. It does what `create_exprun_variant` of ExpRunFlow does, but with an explicit name and inside the flow workspace. It could later move into ExpRunFlow as a general sweep helper, `expand_replications(collection)`.

**Naming:** the run names encode the seeds (`-m<m>-b<b>`). A replication can therefore be found, rerun or inspected by name, and two flows with overlapping seed lists produce the same names for the same replications.

## 5. The result of one replication

### 5.1 Files

Each replication keeps its usual directory `<results>/mrmr2027-run/<base>-m<m>-b<b>/`, with:
- `results.pickle`: the full results, as today, for figures that need trajectories (detection maps, replanning);
- **`metrics.json`** (new): everything the aggregation needs, as plain data, so that aggregating 180 replications does not load 180 pickles with their simulator objects.

`run_mrmr_experiment` writes `metrics.json` for every run, the base runs included. The values below are illustrative:

```json
{
  "run": "clustered-mrmr-m3-b2",
  "base-run": "clustered-mrmr",
  "scenario": "clustered",
  "approach": "mrmr",
  "map-seed": 3,
  "behavior-seed": 2,
  "robots": ["con-1", "con-2", "pio"],
  "scalars": {
    "voi-absolute": 27023.0,
    "voi-estimator": 37990.0,
    "diseased-cells": 500,
    "diseased-found": 270,
    "cells-observed": 2198,
    "messages": 3012,
    "bytes-transmitted": 132500,
    "bytes-delivered": 265000,
    "eps-offered": 4,
    "eps-awarded": 3,
    "eps-completed": 2,
    "bytes-location": 131000,
    "bytes-ep-offer": 500,
    "...": 0,
    "computation-seconds": 8.7
  },
  "per-robot": {
    "con-1": {"voi-absolute": 12000.0, "cells-observed": 900, "diseased-found": 120, "bytes-transmitted": 44000},
    "...": {}
  },
  "series": {
    "timestep": [9, 19, 29, "..."],
    "voi-absolute": ["..."],
    "voi-estimator": ["..."],
    "diseased-found": ["..."],
    "bytes-transmitted-cumulative": ["..."]
  }
}
```

- `scenario` and `approach` are written into the base runs, e.g. `scenario: clustered`, `approach: mrmr`, so that no code parses run names.
- **Seeds of the base runs:** `map-seed` is the map seed of the environment, and `behavior-seed` is `null`, since the base runs use hand-chosen robot seeds.
- **Series:** sampled at the scoring events (every `im_resolution` = 10 timesteps, 100 points per run), on the same timesteps for every replication. They can therefore be averaged pointwise.
- **The scalars** are the final values of the series, plus the counts that have no series.
- `diseased-found` counts the diseased cells observed at least once. With `v_pos = 100` and `v_neg = 1` it can be recomputed from the absolute VoI, but it is the quantity a reader understands.

### 5.2 Determinism check

A replication must be reproducible from its name. The flow can verify this cheaply: rerunning one replication per cell and comparing its `metrics.json` (`scalars` and `series`) with the stored one must give identical values. This is a test, not part of every flow (Section 8).

## 6. Aggregation

A new experiment family, `mrmr2027-aggregate`, with one notebook, `MRMR-Aggregate.ipynb`. It reads the `metrics.json` of the replications listed by its exprun, and writes tidy tables to its data directory. **All later figures read only these tables.**

### 6.1 The tidy tables

All are CSV with one observation per row ("long" format). They load directly into pandas (`pd.read_csv`), and plot with seaborn or matplotlib by grouping on the key columns.

**`replications.csv`:** one row per replication and scalar metric.

| scenario | approach | map_seed | behavior_seed | run | metric | value |
|---|---|---|---|---|---|---|
| clustered | mrmr | 3 | 2 | clustered-mrmr-m3-b2 | voi-absolute | 27023.0 |

**`robots.csv`:** one row per replication, robot and metric. The columns of `replications.csv`, plus `robot` and `role`, where the role is `pioneer`, `contractor`, `lawnmower` or `random-waypoint`. Roles are compared across replications, robot names are not.

**`series.csv`:** one row per replication, series metric and timestep.

| scenario | approach | map_seed | behavior_seed | metric | timestep | value |
|---|---|---|---|---|---|---|

**`summary.csv`:** one row per cell (scenario, approach) and metric. These are the statistics the figures draw:

| column | meaning |
|---|---|
| `n` | number of replications |
| `mean`, `std`, `sem` | mean, standard deviation, standard error |
| `ci95_low`, `ci95_high` | 95% confidence interval of the mean, Student t with `n-1` degrees of freedom |
| `min`, `q25`, `median`, `q75`, `max` | the range and the quartiles |

**`summary-series.csv`:** the same statistics per cell, metric and timestep, for curves with confidence bands.

**`paired.csv`:** one row per scenario, metric and pair of approaches `(a, r)`, for example `(mrmr, mrse)`. It holds the statistics of the per-replication differences `value(a, m, b) - value(r, m, b)`: `n`, `mean`, `ci95_low`, `ci95_high`, `wins` (the fraction of replications where `a` is better), and the p-value of a paired t-test and of a Wilcoxon signed-rank test. **This is what common random numbers buy.** The differences remove the map-to-map variance, which is the largest component, so a difference between approaches can be significant even when their confidence intervals overlap.

**`variance.csv`:** per cell and metric, the split of the variance into the **map** component (the variance of the means over map seeds) and the **behavior** component (the mean of the variances within a map seed). This is only possible with the crossed design, `|M| > 1` and `|B| > 1`. It answers whether more maps or more behavior seeds would narrow the intervals.

### 6.2 Statistics

- **Confidence intervals:** Student t intervals of the mean. With 30 replications per cell this is adequate for the means, even for skewed metrics. Bootstrap intervals (`scipy.stats.bootstrap`) can be added as extra columns if the distributions turn out strongly skewed, for example the number of EPs.
- **The independence assumption** behind the intervals holds across map seeds. Within a map seed, the behavior replications share the map, so the effective sample size lies between `|M|` and `|M| x |B|`. **Recommendation:** `summary.csv` also reports, as `*_maps` columns, the statistics over the per-map means (`n = |M|`), and the figures use those. This is the conservative choice, and with `|B| = 1` the two coincide.

### 6.3 Why tidy tables

- **One format for every figure:** bars with error bars, box plots, curves with bands, scatter plots of map seed against VoI. Each is a group-by on the same tables, so new figures do not need new aggregation code.
- **Incremental growth:** more seeds or more approaches add rows, not columns.
- **Paper tables:** the paper's sync script (`paper-MRMR/scripts/sync_full_results.py`) can generate the results table with "mean ± CI" from `summary.csv`, without unpickling anything.
- **Tool-independence:** the tables are plain CSV, readable by R, a spreadsheet or pgfplots, if the figures are ever drawn in LaTeX.

## 7. Figures

New figure exp/runs in `mrmr2027-figure` read the tables of an aggregate exp/run rather than run results: `source-experiment: mrmr2027-aggregate`, `source-runs: [icc-2027-replicated]`. They share one notebook, `MRMR-Visualize-Replicated.ipynb`, and their `figure-kind` (`comparison`, `series`, `paired`, `communication`, `per-role`) selects the drawing function in `mrmr_graphics.py`:

| Figure | Shows |
|---|---|
| `replicated-comparison` | The summary across the approaches, one panel per metric (absolute VoI, diseased plants found, cells observed). The scenarios are on the x axis, with one bar per approach: the mean with a 95% CI error bar, and the individual replications as dots. |
| `replicated-ranges` | A box-and-whisker ("mustache") version of `replicated-comparison`, in the same layout. The box is the 95% CI of the mean, with a line at the mean; the whiskers are the range of the replications (min to max); the dots are the replications. The box and the whiskers are drawn independently, since with few replications the CI can be wider than the range. |
| `replicated-voi-over-time` | The absolute VoI over time, one panel per scenario: the mean curve per approach, with a 95% CI band and a lighter min–max band. |
| `replicated-paired` | The per-replication differences between the approaches (each pair) with their CI, the zero line, and the fraction of replications won, one panel per scenario. |
| `replicated-communication-cost` | The cumulative transmitted kB of MRMR, mean with a CI band, and the kB per message type with CI error bars. |
| `replicated-per-role` | The VoI per robot by role, mean with CI, one panel per scenario: the pioneer and the contractors, against the random waypoint and lawnmower robots. |

**Shared axes:** bars and curves that are compared share their axis, so that they can be compared visually.
- In `replicated-comparison` and `replicated-ranges`, all the bars or boxes of a metric are in one panel.
- The scenario panels of `replicated-voi-over-time` and `replicated-per-role` share the y axis (`sharey`), and those of `replicated-paired` share the x axis.

The single-run figures (detection maps, replanning snapshots) remain, drawn for one representative replication, e.g. `m = 1`, `b = 1`, or for the base runs.

## 8. Determinism

A replication is only meaningful if it is a function of its seeds:

1. **Bounded, deterministic EP path search (implemented, revised 2026-10-09):** `find_shortest_path_ep(start, end, max_evaluations)` in `exploration_package.py`. It first replaced the wall-clock `maxtime` with a count, which made the runs reproducible. It was then revised, because the count did not bound the work either.
   - **The problem:** like the original time limit, the first version checked the count only after a complete EP order, i.e. after all 4^n lawnmower directions of n EPs. A contractor with 9 or 10 committed EPs therefore evaluated 0.3 to 1 million paths per search, whatever the limit.
     - **Impact:** in the first 20-replication flow, the MRMR runs on the unclustered maps, with 13–18 EPs awarded, took 1–18 minutes of simulation instead of about 5 seconds. Six of the 240 runs took 17–18 minutes each.
     - **Timing:** the time limit of the 2025 code had the same flaw, and hid it as "slow runs".
   - **The fix:** the search is exhaustive only if its whole space fits into the limit, i.e. n! · 4^n ≤ `max_evaluations`. Otherwise `greedy_path_ep` builds the path greedily: from the current position, it takes the remaining EP and lawnmower direction with the smallest cost (the distance to the start of the lawnmower plus its length), and continues from its end.
     - The greedy search is deterministic (ties go to the earlier EP and direction), and takes about 4 n² evaluations.
     - It replaces the earlier exhaustive search cut off within the first EP orders. That search varied the directions of the last EPs while keeping the given commitment order, so it explored hardly any orders.
   - **The limits:** `MRMR_Contractor.BID_EVALUATIONS = 2000` and `REPLAN_EVALUATIONS = 20000`, unchanged. Bidding is exhaustive up to 3 committed EPs (384 combinations) and greedy from 4 (6144). Replanning is exhaustive up to 4 EPs (6144) and greedy from 5 (122880).
   - **Effect on the results:** paths with few EPs are the same optimal paths as before. With more EPs, MRMR's bids and plans differ from the earlier, truncated search, so all the replications were recomputed after the fix. The run that took 1063 s now takes 5.7 s, with the same number of awarded EPs (13).
   - **Exhaustive search:** `None` searches everything. The optimal-path figure uses it (`max-evaluations: null`).
   - **Tests:** `src/test/test_exploration_package.py` checks that a small search is still the unbounded optimum, and that a search with 10 EPs finishes within 2 seconds, covers every EP and is deterministic.
2. **No global random state:** the policies, the environment and the generator already use their own `np.random.default_rng` generators. A test runs one replication twice in the same process, and once in a fresh process, and compares `metrics.json`.
3. **Parallel execution** becomes safe once 1 holds. The replications are independent, so the flow could run them in a process pool, which ExpRunFlow does not do today. Not part of this design.

## 9. Changes summary

| File | Change |
|---|---|
| `src/papers/y2027_mrmr/exploration_package.py` | `find_shortest_path_ep(start, end, max_evaluations)` instead of `maxtime`: exhaustive if the space fits into the limit, otherwise `greedy_path_ep` (Section 8) |
| `src/papers/y2027_mrmr/mrmr_policies.py` | `BID_EVALUATIONS`, `REPLAN_EVALUATIONS` |
| `src/papers/y2027_mrmr/mrmr_metrics.py` | new: `collect_metrics`, `save_metrics`, `load_metrics`, `robot_role` |
| `src/papers/y2027_mrmr/run_experiment.py` | writes `metrics.json`; `run_identification` |
| `data/expruns/mrmr2027-run/*.yaml` | `scenario`, `approach` |
| `src/wbf_flow.py` | `robot_seed`, `replication_name`, `replication_names`, `write_exprun_variant`, `replicated_robots`, `build_mrmr_2027_replicated_entries` |
| `data/expruns/mrmr2027-flow/icc-2027-replicated.yaml` | new (Section 3) |
| `data/expruns/mrmr2027-aggregate/` | new family: `_defaults_mrmr2027-aggregate.yaml`, `icc-2027-replicated.yaml` |
| `src/papers/y2027_mrmr/mrmr_aggregate.py`, `MRMR-Aggregate.ipynb` | new: the tidy tables and statistics (Section 6) |
| `src/papers/y2027_mrmr/mrmr_graphics.py`, `MRMR-Visualize-Replicated.ipynb`, `mrmr2027-figure/replicated-*.yaml` | the figures of Section 7 |
| `src/papers/y2027_mrmr/MRMR-Flow.ipynb` | the `mrmr2027-aggregate` family in the workspace; `icc-2027-replicated` listed |
| `data/expruns/mrmr2027-figure/optimal-ep-path.yaml`, `MRMR-Visualize-OptimalEPPath.ipynb` | `max-evaluations` instead of `max-search-time` |
| `src/test/test_replications.py` | robot seeds; the generated replications; the statistics on a synthetic set |
| `src/test/test_flows.py` | the new notebooks and family |

Not yet done: the paper's results table as mean ± CI from `summary.csv` (`paper-MRMR/scripts/sync_full_results.py`).

## 10. Open questions

1. **How many replications?** 10 map seeds x 3 behavior seeds is a first proposal. Once `variance.csv` shows which component dominates, the seeds should go where the variance is.
2. **The base runs:** should they stay as separate single runs, or be replaced by replication `m = 1, b = 1`, so that the paper shows only replicated results?
3. **Scenarios beyond seeds:** the same machinery replicates parameter sweeps (robot count, budget, Miniberry-200). Should `replications` be generalized to a `factors` section (a factorial design), with `map-seed` and `behavior-seed` as two factors among others?
4. **Reporting the per-map or the per-replication statistics** (Section 6.2): the conservative per-map intervals are proposed for the figures. Reviewers from the networking community typically expect "mean ± 95% CI over N runs", so the text must state what N is.
