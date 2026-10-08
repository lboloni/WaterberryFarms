import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from papers.y2025_mrmr.epmarket import EPAgent, EPM
from papers.y2025_mrmr.exploration_package import ExplorationPackage, ExplorationPackageSet
from papers.y2027_mrmr.mrmr_graphics import build_figure_previews, save_figure
from path_generators import get_path_length


class TestMRMRPrimitives(unittest.TestCase):
    def setUp(self):
        EPM().reset()
        self.market = EPM().epm

    def test_market_assigns_lowest_bid(self):
        owner = EPAgent("owner")
        bidder_1 = EPAgent("bidder-1")
        bidder_2 = EPAgent("bidder-2")
        for agent in [owner, bidder_1, bidder_2]:
            self.market.join(agent)
        offer = owner.offer(ExplorationPackage(0, 4, 0, 4, 2), prize=10)
        bidder_1.bid(offer, 8)
        bidder_2.bid(offer, 6)
        self.market.clearing()
        self.assertEqual(bidder_1.commitments, [])
        self.assertEqual(bidder_2.commitments, [offer])
        self.assertEqual(owner.agreed_deals, [offer])

    def test_reset_removes_previous_market_state(self):
        self.market.join(EPAgent("old"))
        EPM().reset()
        self.assertEqual(EPM().epm.agents, {})

    def test_exploration_package_route_starts_at_requested_point(self):
        packages = ExplorationPackageSet()
        packages.add_ep(ExplorationPackage(1, 3, 1, 3, 1))
        path, labeled_path = packages.find_shortest_path_ep([0, 0], maxtime=1)
        self.assertEqual(path[0].tolist(), [0, 0])
        self.assertIsNone(labeled_path[0]["ep"])
        self.assertGreater(get_path_length(path), 0)

    def test_paper_figures_are_saved_as_pdf_and_png(self):
        class RecordingFigure:
            def __init__(self):
                self.paths = []

            def savefig(self, path):
                self.paths.append(path)

        figure = RecordingFigure()
        save_figure(figure, pathlib.Path("result.pdf"))
        self.assertEqual(
            figure.paths,
            [pathlib.Path("result.pdf"), pathlib.Path("result.png")])

    def test_flow_previews_describe_every_generated_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            comparison = root / "comparison"
            replanning = root / "replanning"
            comparison.mkdir()
            replanning.mkdir()
            (comparison / "comparison.png").touch()
            (replanning / "plans_clustered_con-1_42.0.png").touch()

            experiments = {
                ("figures", "comparison"): {
                    "name": "Clustered comparison",
                    "source-experiment": "runs",
                    "source-runs": ["mrrw", "mrmr"],
                    "data_dir": comparison,
                },
                ("figures", "replanning"): {
                    "name": "Replanning snapshots",
                    "source-experiment": "runs",
                    "source-runs": ["clustered"],
                    "output-filename-prefix": "plans_clustered",
                    "data_dir": replanning,
                },
            }

            class PreviewConfig:
                def get_experiment(
                        self, experiment, run, create_data_dir=True):
                    return experiments[(experiment, run)]

            previews = build_figure_previews(
                "figures", ["comparison", "replanning"],
                config=PreviewConfig())

        self.assertEqual(len(previews), 2)
        self.assertIn("runs/mrrw, runs/mrmr", previews[0]["description"])
        self.assertIn(
            "robot con-1 at t=42.0", previews[1]["description"])
        self.assertIn("Figure: figures/replanning", previews[1]["description"])


if __name__ == "__main__":
    unittest.main()
