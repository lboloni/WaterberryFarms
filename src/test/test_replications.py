import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from exp_run_config import Config
from papers.y2027_mrmr.mrmr_aggregate import aggregate_tables, confidence_interval, describe
from wbf_flow import build_mrmr_2027_flow_entries, replication_names, robot_seed

EXPRUNS = pathlib.Path(__file__).resolve().parents[2] / "data" / "expruns"


class TestRobotSeeds(unittest.TestCase):
    def test_robot_seeds_are_stable_and_distinct(self):
        self.assertEqual(robot_seed(1, 0), robot_seed(1, 0))
        seeds = {robot_seed(b, i) for b in range(1, 4) for i in range(3)}
        self.assertEqual(len(seeds), 9)


class TestReplicationGeneration(unittest.TestCase):
    def setUp(self):
        Config.PROJECTNAME = "WaterBerryFarms"
        self.directory = tempfile.TemporaryDirectory()
        self.expruns = pathlib.Path(self.directory.name, "expruns")
        shutil.copytree(EXPRUNS, self.expruns)
        self.previous = Config().get_exprun_path()
        Config().set_exprun_path(self.expruns)

    def tearDown(self):
        Config().set_exprun_path(self.previous)
        self.directory.cleanup()

    def load(self, family, run):
        with open(self.expruns / family / f"{run}.yaml") as f:
            return yaml.safe_load(f)

    def test_replications_are_generated_with_their_seeds(self):
        entries = build_mrmr_2027_flow_entries("mrmr2027-flow", "icc-2027-replicated", "discard-old")
        aggregate = Config().get_experiment("mrmr2027-aggregate", "icc-2027-replicated", create_data_dir=False)
        runs = replication_names(aggregate)
        self.assertEqual(len(runs), 6 * len(aggregate["map-seeds"]) * len(aggregate["behavior-seeds"]))
        # the phases: the environment variants, the replications, the aggregation, the figures
        names = [entry["name"].split()[0] for entry in entries]
        environments = 2 * len(aggregate["map-seeds"])
        self.assertEqual(names, ["Precompute"] * environments + ["Run"] * len(runs) + ["Aggregate"] + ["Figure"] * 6)
        self.assertEqual([entry["run"] for entry in entries if entry["name"].startswith("Run")], runs)

        environment = self.load("environment", "mrmr-generated-clustered-100-m1")
        self.assertEqual(environment["tylcv-generated-seed"], 1)
        mrmr = self.load("mrmr2027-run", "clustered-mrmr-m1-b2")
        self.assertEqual((mrmr["base-run"], mrmr["behavior-seed"], mrmr["run_environment"]),
                         ("clustered-mrmr", 2, "mrmr-generated-clustered-100-m1"))
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in mrmr["robots"]],
                         [robot_seed(2, i) for i in range(3)])
        # common random numbers: robot i of every approach has the same seed
        mrrw = self.load("mrmr2027-run", "clustered-mrrw-m1-b2")
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in mrrw["robots"]],
                         [robot_seed(2, i) for i in range(3)])
        # the lawnmowers have no seed
        mrse = self.load("mrmr2027-run", "clustered-mrse-m1-b2")
        self.assertTrue(all("seed" not in r["exp-policy-extra-parameters"] for r in mrse["robots"]))
        # the variants resolve, with the defaults of their family
        exp = Config().get_experiment("mrmr2027-run", "clustered-mrmr-m1-b2", create_data_dir=False)
        self.assertEqual(exp["communication-rounds"], 3)


def synthetic_metrics():
    """Two approaches, two maps, three behavior seeds: value = 100 * map + 10 * behavior (+ 5 for a)"""
    metrics = []
    for approach, offset in [("a", 5), ("r", 0)]:
        for m in [1, 2]:
            for b in [1, 2, 3]:
                value = 100 * m + 10 * b + offset
                metrics.append({
                    "run": f"x-{approach}-m{m}-b{b}", "base-run": f"x-{approach}", "scenario": "x",
                    "approach": approach, "map-seed": m, "behavior-seed": b, "robots": ["r1"],
                    "scalars": {"voi-absolute": value},
                    "per-robot": {"r1": {"role": "pioneer", "voi-absolute": value}},
                    "series": {"timestep": [9, 19], "voi-absolute": [value / 2, value]}})
    return metrics


class TestAggregation(unittest.TestCase):
    def test_summary_paired_and_variance(self):
        tables = aggregate_tables(synthetic_metrics(), ["a", "r"])
        self.assertEqual(len(tables["replications"]), 12)
        row = tables["summary"].query("approach == 'a' and metric == 'voi-absolute'").iloc[0]
        values = [100 * m + 10 * b + 5 for m in [1, 2] for b in [1, 2, 3]]
        self.assertEqual(row["n"], 6)
        self.assertAlmostEqual(row["mean"], np.mean(values))
        low, high = stats.t.interval(0.95, 5, loc=np.mean(values), scale=stats.sem(values))
        self.assertAlmostEqual(row["ci95_low"], low)
        self.assertAlmostEqual(row["ci95_high"], high)
        # with two maps, the figures use the interval over the two per-map means
        self.assertEqual(row["n_maps"], 2)
        self.assertEqual(confidence_interval(row), (row["ci95_low_maps"], row["ci95_high_maps"]))
        # the paired difference is exactly 5 in every replication
        paired = tables["paired"].iloc[0]
        self.assertEqual((paired["approach"], paired["reference"], paired["mean"], paired["fraction_greater"]),
                         ("a", "r", 5.0, 1.0))
        # map variance: of the per-map means 120, 220; behavior variance: of 10, 20, 30 within a map
        variance = tables["variance"].query("approach == 'a'").iloc[0]
        self.assertAlmostEqual(variance["variance_map"], np.var([120, 220], ddof=1))
        self.assertAlmostEqual(variance["variance_behavior"], np.var([10, 20, 30], ddof=1))
        series = tables["summary-series"].query("approach == 'r' and timestep == 9").iloc[0]
        self.assertAlmostEqual(series["mean"], np.mean([(100 * m + 10 * b) / 2 for m in [1, 2] for b in [1, 2, 3]]))

    def test_a_single_value_has_no_interval(self):
        d = describe([3.0])
        self.assertEqual((d["n"], d["mean"]), (1, 3.0))
        self.assertTrue(np.isnan(d["ci95_low"]))


if __name__ == "__main__":
    unittest.main()
