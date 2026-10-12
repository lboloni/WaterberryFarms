"""The replications of the MAREX comparisons: the project's appliers (what a map seed and a behavior seed
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
from wbf_flow import build_marex_2027_flow_entries

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
        entries = build_marex_2027_flow_entries("marex2027-flow", "icc-2027-replicated", "discard-old")
        aggregate = Config().get_experiment("marex2027-aggregate", "icc-2027-replicated", create_data_dir=False)
        maps = aggregate["factors"]["map-seed"]["values"]
        runs = replication_names(aggregate)
        self.assertEqual(len(runs), len(aggregate["runs"]) * len(maps) * len(aggregate["factors"]["behavior-seed"]["values"]))
        flow = Config().get_experiment("marex2027-flow", "icc-2027-replicated", create_data_dir=False)
        # the phases: the environment variants (2 sizes x 2 scenarios x the map seeds), the replications,
        # the aggregation, the figures
        names = [entry["name"].split()[0] for entry in entries]
        self.assertEqual(names, ["Prepare"] * (4 * len(maps)) + ["Run"] * len(runs) + ["Aggregate"]
                         + ["Figure"] * len(flow["figures"]))
        self.assertEqual([entry["run"] for entry in entries if entry["name"].startswith("Run")], runs)

        environment = self.load("environment", "marex-generated-clustered-200-m3")
        self.assertEqual((environment["tylcv-generated-seed"], environment["typename"]), (3, "Miniberry-200"))
        marex = self.load("marex2027-run", "clustered-marex-m3-b2")
        self.assertEqual(marex["replication"], {"base-run": "clustered-marex", "factors": {"map-seed": 3, "behavior-seed": 2},
                         "labels": {"map-size": 100, "team-size": 3, "scenario": "clustered", "approach": "marex"}})
        self.assertEqual(marex["run_environment"], "marex-generated-clustered-100-m3")
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in marex["robots"]],
                         [derive_seed(2, i) for i in range(3)])
        # common random numbers: robot i of every approach has the same seed
        mrrw = self.load("marex2027-run", "unclustered-200-mrrw-m3-b2")
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in mrrw["robots"]],
                         [derive_seed(2, i) for i in range(3)])
        # the balanced lawnmowers have no seed, the balanced confidence-guided robots have one each
        bmrse = self.load("marex2027-run", "clustered-bmrse-m3-b2")
        self.assertTrue(all("seed" not in r["exp-policy-extra-parameters"] for r in bmrse["robots"]))
        bmrcg = self.load("marex2027-run", "clustered-bmrcg-m3-b2")
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in bmrcg["robots"]],
                         [derive_seed(2, i) for i in range(3)])
        # the variants resolve, with the defaults of their family
        exp = Config().get_experiment("marex2027-run", "clustered-200-marex-m3-b2", create_data_dir=False)
        self.assertEqual(exp["communication-rounds"], 3)
        # the teams of 6: one pioneer and five contractors, every robot with its own seed
        six = self.load("marex2027-run", "clustered-200-6-marex-m3-b2")
        self.assertEqual(six["replication"]["labels"], {"map-size": 200, "team-size": 6, "scenario": "clustered",
                                                        "approach": "marex"})
        self.assertEqual([r["name"] for r in six["robots"]], ["pio"] + [f"con-{k}" for k in range(1, 6)])
        self.assertEqual([r["exp-policy-extra-parameters"]["seed"] for r in six["robots"]],
                         [derive_seed(2, i) for i in range(6)])

    def test_the_paper_flow_runs_the_examples_before_the_replications(self):
        entries = build_marex_2027_flow_entries("marex2027-flow", "icc-2027-paper", "discard-old")
        flow = Config().get_experiment("marex2027-flow", "icc-2027-paper", create_data_dir=False)
        aggregate = Config().get_experiment("marex2027-aggregate", flow["aggregate"], create_data_dir=False)
        maps = aggregate["factors"]["map-seed"]["values"]
        environments = {Config().get_experiment("marex2027-run", run, create_data_dir=False)["run_environment"]
                        for run in flow["runs"]}
        names = [entry["name"].split()[0] for entry in entries]
        self.assertEqual(names, ["Precompute"] * len(environments) + ["Run"] * len(flow["runs"])
                         + ["Figure"] * len(flow["figures"]) + ["Prepare"] * (4 * len(maps))
                         + ["Run"] * len(replication_names(aggregate)) + ["Aggregate"]
                         + ["Figure"] * len(flow["aggregate-figures"]))
        figures = [entry["run"] for entry in entries if entry["name"].startswith("Figure")]
        self.assertEqual(figures, flow["figures"] + flow["aggregate-figures"])

    def test_a_partial_paper_flow_runs_three_replications_on_three_maps(self):
        entries = build_marex_2027_flow_entries("marex2027-flow", "icc-2027-paper", "discard-old", replications=3)
        flow = Config().get_experiment("marex2027-flow", "icc-2027-paper", create_data_dir=False)
        aggregate = Config().get_experiment("marex2027-aggregate", flow["aggregate"], create_data_dir=False)
        self.assertEqual(aggregate["replications"], 3)
        replications = [entry["run"] for entry in entries if entry["name"].startswith("Run")][len(flow["runs"]):]
        self.assertEqual(len(replications), 3 * len(aggregate["runs"]))
        self.assertEqual(replications[:3], ["clustered-marex-m1-b1", "clustered-marex-m2-b1", "clustered-marex-m3-b1"])
        # the map variants of the three maps only, for the 4 generated maps
        self.assertEqual(sum(entry["name"].startswith("Prepare") for entry in entries), 4 * 3)


class TestReproducibility(unittest.TestCase):
    def test_an_marex_run_is_a_function_of_its_seeds(self):
        """Two runs of the same exp/run give the same metrics (no wall-clock or unseeded randomness)"""
        from exprunflow.replication import check_reproducible
        from papers.y2027_marex.run_experiment import run_marex_experiment
        Config.PROJECTNAME = "WaterBerryFarms"
        same, first, second = check_reproducible("marex2027-run", "unclustered-marex", run_marex_experiment,
                                                 ignore=("computation-seconds",))
        self.assertTrue(same, (first["scalars"], second["scalars"]))


if __name__ == "__main__":
    unittest.main()
