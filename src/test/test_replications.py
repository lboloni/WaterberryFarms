"""The replications of the MRMR comparisons: the project's appliers (what a map seed and a behavior seed
change), on top of the ExpRunFlow library (exprunflow.replication), whose own tests cover the mechanism
and the statistics."""

import pathlib
import shutil
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from exp_run_config import Config
from exprunflow.replication import derive_seed, replication_names
from wbf_flow import build_mrmr_2027_flow_entries

EXPRUNS = pathlib.Path(__file__).resolve().parents[2] / "data" / "expruns"


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
        maps = aggregate["factors"]["map-seed"]["values"]
        runs = replication_names(aggregate)
        self.assertEqual(len(runs), len(aggregate["runs"]) * len(maps) * len(aggregate["factors"]["behavior-seed"]["values"]))
        flow = Config().get_experiment("mrmr2027-flow", "icc-2027-replicated", create_data_dir=False)
        # the phases: the environment variants (2 sizes x 2 scenarios x the map seeds), the replications,
        # the aggregation, the figures
        names = [entry["name"].split()[0] for entry in entries]
        self.assertEqual(names, ["Prepare"] * (4 * len(maps)) + ["Run"] * len(runs) + ["Aggregate"]
                         + ["Figure"] * len(flow["figures"]))
        self.assertEqual([entry["run"] for entry in entries if entry["name"].startswith("Run")], runs)

        environment = self.load("environment", "mrmr-generated-clustered-200-m3")
        self.assertEqual((environment["tylcv-generated-seed"], environment["typename"]), (3, "Miniberry-200"))
        mrmr = self.load("mrmr2027-run", "clustered-mrmr-m3-b2")
        self.assertEqual(mrmr["replication"], {"base-run": "clustered-mrmr", "factors": {"map-seed": 3, "behavior-seed": 2},
                         "labels": {"map-size": 100, "scenario": "clustered", "approach": "mrmr"}})
        self.assertEqual(mrmr["run_environment"], "mrmr-generated-clustered-100-m3")
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in mrmr["robots"]],
                         [derive_seed(2, i) for i in range(3)])
        # common random numbers: robot i of every approach has the same seed
        mrrw = self.load("mrmr2027-run", "unclustered-200-mrrw-m3-b2")
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in mrrw["robots"]],
                         [derive_seed(2, i) for i in range(3)])
        # the lawnmowers have no seed
        mrse = self.load("mrmr2027-run", "clustered-mrse-m3-b2")
        self.assertTrue(all("seed" not in r["exp-policy-extra-parameters"] for r in mrse["robots"]))
        # the variants resolve, with the defaults of their family
        exp = Config().get_experiment("mrmr2027-run", "clustered-200-mrmr-m3-b2", create_data_dir=False)
        self.assertEqual(exp["communication-rounds"], 3)


class TestReproducibility(unittest.TestCase):
    def test_an_mrmr_run_is_a_function_of_its_seeds(self):
        """Two runs of the same exp/run give the same metrics (no wall-clock or unseeded randomness)"""
        from exprunflow.replication import check_reproducible
        from papers.y2027_mrmr.run_experiment import run_mrmr_experiment
        Config.PROJECTNAME = "WaterBerryFarms"
        same, first, second = check_reproducible("mrmr2027-run", "unclustered-mrmr", run_mrmr_experiment,
                                                 ignore=("computation-seconds",))
        self.assertTrue(same, (first["scalars"], second["scalars"]))


if __name__ == "__main__":
    unittest.main()
