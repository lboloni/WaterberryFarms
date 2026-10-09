"""
mrmr_aggregate.py

The aggregation of the replications of the MRMR comparisons into tidy tables (one observation per row),
with the statistics from which the figures draw means, confidence intervals and ranges.
See DESIGN-MultiSeedEvaluation.md, Section 6.
"""

import itertools
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from exp_run_config import Config

from .mrmr_metrics import load_metrics

KEYS = ["scenario", "approach", "map_seed", "behavior_seed", "run"]
CELL = ["scenario", "approach"]
TABLES = ["replications", "robots", "series", "summary", "summary-series", "paired", "variance"]


def load_replications(aggregate, config=None):
    """The metrics of the replications of an aggregate exp/run"""
    from wbf_flow import replication_names
    if config is None:
        config = Config()
    metrics = []
    for run in replication_names(aggregate):
        exp = config.get_experiment(aggregate["source-experiment"], run, create_data_dir=False)
        metrics.append(load_metrics(exp["data_dir"]))
    return metrics


def identification(m):
    return {"scenario": m["scenario"], "approach": m["approach"], "map_seed": m["map-seed"],
            "behavior_seed": m["behavior-seed"], "run": m["run"]}


def tidy_tables(metrics):
    """The replications, robots and series tables of a list of metrics"""
    replications, robots, series = [], [], []
    for m in metrics:
        ident = identification(m)
        for metric, value in m["scalars"].items():
            replications.append({**ident, "metric": metric, "value": value})
        for robot, values in m["per-robot"].items():
            for metric, value in values.items():
                if metric != "role":
                    robots.append({**ident, "robot": robot, "role": values["role"], "metric": metric, "value": value})
        timesteps = m["series"]["timestep"]
        for metric, values in m["series"].items():
            if metric != "timestep":
                series += [{**ident, "metric": metric, "timestep": t, "value": v} for t, v in zip(timesteps, values)]
    return pd.DataFrame(replications), pd.DataFrame(robots), pd.DataFrame(series)


def describe(values):
    """The statistics of a sample: n, mean, std, sem, the 95% Student t confidence interval of the mean,
    and the range and quartiles. With n = 1 the spread statistics are NaN."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else np.nan
    sem = std / np.sqrt(n) if n > 1 else np.nan
    half = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else np.nan
    return {"n": n, "mean": mean, "std": std, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half,
            "min": float(np.min(values)), "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values))}


def summarize(table, keys):
    """The statistics per group of keys, over the replications, and (the _maps columns) over the per-map
    means, which are independent even when several behavior seeds share a map"""
    rows = []
    for group, frame in table.groupby(keys, sort=False):
        row = dict(zip(keys, group))
        row.update(describe(frame["value"]))
        per_map = frame.groupby("map_seed")["value"].mean()
        maps = describe(per_map)
        row.update({"n_maps": maps["n"], "ci95_low_maps": maps["ci95_low"], "ci95_high_maps": maps["ci95_high"]})
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(replications, approaches):
    """The statistics of the per-replication differences value(a) - value(r) for every pair of approaches,
    matched on (map seed, behavior seed): the common random numbers make the comparison paired"""
    rows = []
    for (scenario, metric), frame in replications.groupby(["scenario", "metric"], sort=False):
        wide = frame.pivot_table(index=["map_seed", "behavior_seed"], columns="approach", values="value")
        for a, r in itertools.combinations([x for x in approaches if x in wide.columns], 2):
            diff = (wide[a] - wide[r]).dropna()
            if diff.empty:
                continue
            row = {"scenario": scenario, "metric": metric, "approach": a, "reference": r}
            row.update(describe(diff))
            row["fraction_greater"] = float(np.mean(diff > 0))
            row["fraction_less"] = float(np.mean(diff < 0))
            row["p_ttest"] = float(stats.ttest_rel(wide.loc[diff.index, a], wide.loc[diff.index, r]).pvalue) \
                if len(diff) > 1 and np.any(diff != diff.iloc[0]) else np.nan
            row["p_wilcoxon"] = float(stats.wilcoxon(diff).pvalue) if len(diff) > 1 and np.any(diff != 0) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def variance_components(replications):
    """Per cell and metric, the variance between maps (of the per-map means) and the variance of the
    behavior (the mean of the within-map variances). NaN unless there are several map and behavior seeds."""
    rows = []
    for (scenario, approach, metric), frame in replications.groupby(CELL + ["metric"], sort=False):
        by_map = frame.groupby("map_seed")["value"]
        maps, behaviors = by_map.ngroups, int(by_map.size().min())
        rows.append({"scenario": scenario, "approach": approach, "metric": metric,
                     "map_seeds": maps, "behavior_seeds": behaviors,
                     "variance_map": float(by_map.mean().var(ddof=1)) if maps > 1 else np.nan,
                     "variance_behavior": float(by_map.var(ddof=1).mean()) if behaviors > 1 else np.nan})
    return pd.DataFrame(rows)


def aggregate_tables(metrics, approaches):
    """All the tables of Section 6.1, by name"""
    replications, robots, series = tidy_tables(metrics)
    return {
        "replications": replications,
        "robots": robots,
        "series": series,
        "summary": summarize(replications, CELL + ["metric"]),
        "summary-series": summarize(series, CELL + ["metric", "timestep"]),
        "paired": paired_differences(replications, approaches),
        "variance": variance_components(replications),
    }


def save_tables(tables, directory):
    for name, table in tables.items():
        table.to_csv(pathlib.Path(directory, f"{name}.csv"), index=False)


def load_tables(directory):
    return {name: pd.read_csv(pathlib.Path(directory, f"{name}.csv")) for name in TABLES
            if pathlib.Path(directory, f"{name}.csv").exists()}


def confidence_interval(row):
    """The confidence interval a figure shows: over the per-map means if there are several maps
    (conservative, DESIGN-MultiSeedEvaluation.md, Section 6.2), otherwise over the replications"""
    if row["n_maps"] > 1:
        return row["ci95_low_maps"], row["ci95_high_maps"]
    return row["ci95_low"], row["ci95_high"]
