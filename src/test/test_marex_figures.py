import pathlib
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from papers.y2027_marex.marex_graphics import (agentwise_voi, choose_lifecycle, detection_streak, market_lifecycles,
    show_agentwise_voi, show_communication_cost, show_comparative_voi, show_market_sequence)


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


def send(timestep, round, sender, destination, content):
    """A traffic record of the communication medium"""
    return {"timestep": timestep, "round": round, "sender": sender, "destination": destination,
            "type": content["type"], "bytes": 50, "recipients": 2 if destination is None else 1, "content": content}


def observation(robot, time, value):
    return {"robot": robot, "time": time, "TYLCV": {"value": value}}


EP = {"x_min": 10, "x_max": 20, "y_min": 30, "y_max": 40, "step": 2}
# two offers of the Pioneer: pio-0 gets one bid and is completed, pio-1 gets two bids and is completed
MARKET = {"robot-names": ["pio", "con-1", "con-2"], "communication-traffic": [
    send(1, 0, "con-1", None, {"type": "location", "x": 0, "y": 0, "time": 1}),
    send(5, 0, "pio", None, {"type": "ep-offer", "offer-id": "pio-0", "ep": EP, "prize": 8}),
    send(5, 1, "con-2", "pio", {"type": "ep-bid", "offer-id": "pio-0", "value": 8}),
    send(5, 2, "pio", "con-2", {"type": "ep-award", "offer-id": "pio-0", "price": 8}),
    send(9, 0, "pio", None, {"type": "ep-offer", "offer-id": "pio-1", "ep": EP, "prize": 6}),
    send(9, 1, "con-1", "pio", {"type": "ep-bid", "offer-id": "pio-1", "value": 6}),
    send(9, 1, "con-2", "pio", {"type": "ep-bid", "offer-id": "pio-1", "value": 6}),
    send(9, 2, "pio", "con-1", {"type": "ep-award", "offer-id": "pio-1", "price": 6}),
    send(20, 0, "con-2", "pio", {"type": "ep-completed", "offer-id": "pio-0", "real-value": 0}),
    send(30, 0, "con-1", "pio", {"type": "ep-completed", "offer-id": "pio-1", "real-value": 0})],
    # detections (TYLCV 0) of the Pioneer at timesteps 2, 3 and 6, 7, 8
    "observations": [[observation("pio", t, 0.0 if t in (2, 3, 6, 7, 8) else 1.0),
                      observation("con-1", t, 0.0)] for t in range(10)]}


class TestMarketSequence(unittest.TestCase):
    def test_the_messages_are_linked_by_offer_id(self):
        lifecycles = market_lifecycles(MARKET)
        self.assertEqual(sorted(lifecycles), ["pio-0", "pio-1"])
        lifecycle = lifecycles["pio-1"]
        self.assertEqual([bid["sender"] for bid in lifecycle["bids"]], ["con-1", "con-2"])
        self.assertEqual(lifecycle["award"]["destination"], "con-1")
        self.assertEqual(lifecycle["completed"]["timestep"], 30)

    def test_the_first_offer_with_several_bids_is_chosen(self):
        self.assertEqual(choose_lifecycle(market_lifecycles(MARKET))["offer-id"], "pio-1")

    def test_only_the_pioneers_offers_are_chosen(self):
        # a contractor's earlier offer with several bids and a completion is not a candidate
        traffic = [send(4, 0, "con-1", None, {"type": "ep-offer", "offer-id": "con-0", "ep": EP, "prize": 8}),
                   send(4, 1, "con-2", "con-1", {"type": "ep-bid", "offer-id": "con-0", "value": 3}),
                   send(4, 1, "pio", "con-1", {"type": "ep-bid", "offer-id": "con-0", "value": 3}),
                   send(4, 2, "con-1", "con-2", {"type": "ep-award", "offer-id": "con-0", "price": 3}),
                   send(8, 0, "con-2", "con-1", {"type": "ep-completed", "offer-id": "con-0", "real-value": 0})]
        market = dict(MARKET, **{"communication-traffic": traffic + MARKET["communication-traffic"]})
        self.assertEqual(choose_lifecycle(market_lifecycles(market))["offer-id"], "pio-1")

    def test_traffic_without_contents_is_rejected(self):
        traffic = [{k: v for k, v in record.items() if k != "content"} for record in MARKET["communication-traffic"]]
        with self.assertRaises(ValueError):
            market_lifecycles({"communication-traffic": traffic})

    def test_the_streak_is_the_last_one_of_the_robot_before_the_offer(self):
        self.assertEqual(detection_streak(MARKET, "pio", 9), (6, 8, 3))
        self.assertEqual(detection_streak(MARKET, "pio", 5), (2, 3, 2))
        self.assertIsNone(detection_streak(MARKET, "pio", 2))

    def test_the_chart_is_saved_as_pdf_and_png(self):
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = show_market_sequence(TestDirectory(pathlib.Path(directory)), MARKET, offer_id="pio-0",
                                             output_filename="market.pdf")
            self.assertEqual(lifecycle["offer-id"], "pio-0")
            self.assertEqual(sorted(path.name for path in pathlib.Path(directory).iterdir()), ["market.pdf", "market.png"])


if __name__ == "__main__":
    unittest.main()
