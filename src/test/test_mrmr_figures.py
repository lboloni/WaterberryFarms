import pathlib
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from papers.y2027_mrmr.mrmr_graphics import agentwise_voi, show_agentwise_voi, show_communication_cost, show_comparative_voi


def results_with_credits(rows, final):
    """Results of a two-robot run with one scoring event at the last timestep"""
    return {"simulation-timestep": 3, "robot-names": ["con-1", "pio"],
            "score-events": [{"timestep": 2, "score": dict(final, **{"voi-credits": rows})}],
            "score": final}


COMMUNICATION = {"communication-summary": {
    "bytes-per-timestep": [0, 45, 90, 45],
    "per-type": {"location": {"messages": 3, "bytes-transmitted": 135, "bytes-delivered": 270},
                 "ep-offer": {"messages": 1, "bytes-transmitted": 45, "bytes-delivered": 90}}}}

RESULTS = results_with_credits(
    [{"variant": "voi-absolute", "robot": "pio", "timestep": 0, "value": 100.0},
     {"variant": "voi-absolute", "robot": "con-1", "timestep": 1, "value": 1.0},
     {"variant": "voi-absolute", "robot": "pio", "timestep": 2, "value": 1.0},
     {"variant": "voi-estimator", "robot": "pio", "timestep": 0, "value": 40.0},
     {"variant": "voi-estimator", "robot": None, "timestep": 2, "value": -5.0}],
    {"voi-absolute": 102.0, "voi-estimator": -12.0})


class TestDirectory:
    """Stands in for an exprun: the figures go into its data dir"""
    def __init__(self, directory):
        self.directory = directory

    def data_dir(self):
        return self.directory


class TestAgentwiseVoI(unittest.TestCase):
    def test_absolute_credits_per_robot_add_up_to_the_total(self):
        values = agentwise_voi(RESULTS, "voi-absolute")
        self.assertEqual(values, {"con-1": 1.0, "pio": 101.0})
        self.assertEqual(sum(values.values()), RESULTS["score"]["voi-absolute"])

    def test_estimator_update_terms_get_their_own_bar(self):
        self.assertEqual(agentwise_voi(RESULTS, "voi-estimator"), {"con-1": 0.0, "pio": 40.0, "Estimator": -5.0})


class TestVoIFigures(unittest.TestCase):
    def test_figures_are_saved_as_pdf_and_png(self):
        with tempfile.TemporaryDirectory() as directory:
            exp_dest = TestDirectory(pathlib.Path(directory))
            show_agentwise_voi(exp_dest, "run", RESULTS, ["#E69F00", "#56B4E9"], output_filename="agent.pdf")
            show_comparative_voi(exp_dest, "env", {"a": RESULTS, "b": RESULTS}, {"a": "A"}, ["#CC6666"],
                                 output_filename="comparison.pdf")
            show_communication_cost(exp_dest, "cost", {"a": COMMUNICATION, "b": COMMUNICATION}, {"a": "A"},
                                    ["#66CC99"], output_filename="cost.pdf")
            self.assertEqual(sorted(path.name for path in pathlib.Path(directory).iterdir()),
                             ["agent.pdf", "agent.png", "comparison.pdf", "comparison.png", "cost.pdf", "cost.png"])


if __name__ == "__main__":
    unittest.main()
