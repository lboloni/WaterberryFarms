import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium, traffic_summary
from papers.y2027_marex.epmarket import EPAgent, EPOffer
from papers.y2027_marex.exploration_package import ExplorationPackage
from papers.y2027_marex.marex_policies import MAREX_Contractor, MAREX_Pioneer
from robot import Robot
from wbf_simulate import simulate_1day


class HealthyEnvironment:
    """Every observation is healthy, so the pioneer never offers on its own"""
    def __init__(self):
        self.time = 0

    def get_observation(self, position):
        return {"x": position[0], "y": position[1], "time": position[2],
                "TYLCV": {"x": position[0], "y": position[1], "time": position[2], "value": 1.0}}


class SizedEstimator:
    def __init__(self):
        self.width, self.height = 31, 31

    def add_observation(self, observation):
        pass

    def proceed(self, delta_t):
        pass


class ZeroEvaluator:
    def score(self, environment, estimator):
        return 0


def run_market(communication_rounds, timesteps=60):
    """A pioneer and two contractors; after timestep 0, the pioneer offers two exploration packages.
    Returns the policies, the contractors' commitments at the end of every timestep, the medium and the results."""
    environment = HealthyEnvironment()
    medium = PerfectCommunicationMedium(environment)
    policies = {"pio": MAREX_Pioneer({"policy-name": "pioneer", "seed": 1, "budget": timesteps}, None),
                "con-1": MAREX_Contractor({"policy-name": "contractor-1", "seed": 2, "budget": timesteps}, None),
                "con-2": MAREX_Contractor({"policy-name": "contractor-2", "seed": 3, "budget": timesteps}, None)}
    robots = []
    for name, policy in policies.items():
        robot = Robot(name, 0, 0, 0)
        robot.assign_policy(policy)
        medium.add_robot(robot)
        robots.append(robot)
    commitments = []

    def after_timestep(results, environment, robots, estimator, evaluator):
        if results["simulation-timestep"] == 1:
            policies["pio"].epagent.offer(ExplorationPackage(2, 6, 2, 6, step=2), prize=10)
            policies["pio"].epagent.offer(ExplorationPackage(10, 14, 10, 14, step=2), prize=10)
        commitments.append({name: sorted(policies[name].epagent.commitments) for name in ["con-1", "con-2"]})

    results = simulate_1day(environment=environment, robots=robots, estimator=SizedEstimator(),
                  evaluator=ZeroEvaluator(), timesteps=timesteps, estimator_interval=10,
                  communication=medium, communication_rounds=communication_rounds,
                  after_timestep=after_timestep)
    return policies, commitments, medium, results


class TestEPAgent(unittest.TestCase):
    def test_lowest_bid_wins_and_ties_go_to_name_order(self):
        owner = EPAgent("owner")
        offer = owner.offer(ExplorationPackage(0, 4, 0, 4, 2), prize=10)
        self.assertEqual(offer.offer_id, "owner-0")
        owner.receive_bid(offer.offer_id, "b", 6)
        owner.receive_bid(offer.offer_id, "a", 6)
        owner.receive_bid(offer.offer_id, "c", 8)
        awarded = owner.clear(offer.offer_id)
        self.assertEqual((awarded.assigned_to_name, awarded.bid_prize), ("a", 6))
        self.assertEqual(list(owner.agreed_deals), ["owner-0"])

    def test_offer_without_bids_is_declined(self):
        owner = EPAgent("owner")
        offer = owner.offer(ExplorationPackage(0, 4, 0, 4, 2), prize=10)
        self.assertIsNone(owner.clear(offer.offer_id))
        self.assertEqual(owner.declined_offers, [offer])


class TestMarketProtocol(unittest.TestCase):
    def test_auction_completes_within_one_timestep_with_three_rounds(self):
        policies, commitments, _, _ = run_market(communication_rounds=3)
        # offered after timestep 0, broadcast, bid and awarded in the rounds of timestep 1
        self.assertEqual(commitments[0], {"con-1": [], "con-2": []})
        # both contractors start at the home base and bid the same on the first package; the tie goes to
        # con-1, first in name order. The first package lies on the way to the second, so con-1, which
        # counts its bid on the first, has the lower marginal cost for the second as well
        self.assertEqual(commitments[1], {"con-1": ["pioneer-0", "pioneer-1"], "con-2": []})

    def test_completion_settles_the_money_on_both_sides(self):
        policies, commitments, _, _ = run_market(communication_rounds=3)
        pioneer, contractor = policies["pio"].epagent, policies["con-1"].epagent
        # the first package is completed when con-1 moves on to the second one; the price is the winning
        # bid, in timesteps of exploration budget, not the prize of the offer
        self.assertEqual([deal.offer_id for deal in pioneer.terminated_deals], ["pioneer-0"])
        price = pioneer.terminated_deals[0].bid_prize
        self.assertNotEqual(price, 10)
        self.assertEqual(contractor.money, price)
        self.assertEqual(pioneer.money, 0 - price)

    def test_auction_spreads_over_timesteps_with_one_round(self):
        policies, commitments, _, _ = run_market(communication_rounds=1)
        # offer in timestep 1, bids in timestep 2, award in timestep 3
        self.assertEqual(commitments[2], {"con-1": [], "con-2": []})
        self.assertEqual(commitments[3], {"con-1": ["pioneer-0", "pioneer-1"], "con-2": []})

    def test_robots_hold_no_references_to_each_other(self):
        policies, _, _, _ = run_market(communication_rounds=3)
        others = {id(policy) for policy in policies.values()} | {id(policy.epagent) for policy in policies.values()}
        for policy in policies.values():
            own = {id(policy), id(policy.epagent)}
            for value in list(vars(policy).values()) + list(vars(policy.epagent).values()):
                self.assertNotIn(id(value), others - own)


def cell_counts(observations):
    """{cell: count} of a list of observations"""
    counts = {}
    for observation in observations:
        cell = (observation["x"], observation["y"])
        counts[cell] = counts.get(cell, 0) + 1
    return counts


class TestLocationBroadcast(unittest.TestCase):
    def test_every_robot_broadcasts_its_location_once_per_timestep(self):
        timesteps = 60
        _, _, medium, _ = run_market(communication_rounds=3, timesteps=timesteps)
        locations = [record for record in medium.traffic if record["type"] == "location"]
        # nothing is observed before timestep 0, so the first broadcast is in timestep 1
        for name in ["con-1", "con-2", "pio"]:
            sent = [record for record in locations if record["sender"] == name]
            self.assertEqual([record["timestep"] for record in sent], list(range(1, timesteps)))
            self.assertTrue(all(record["round"] == 0 and record["destination"] is None
                                and record["recipients"] == 2 for record in sent))

    def test_explored_map_holds_own_cells_and_the_others_cells_one_timestep_late(self):
        timesteps = 60
        policies, _, _, results = run_market(communication_rounds=3, timesteps=timesteps)
        observations = [observation for step in results["observations"] for observation in step]
        for name, policy in policies.items():
            expected = cell_counts([observation for observation in observations
                if observation["robot"] == name or observation["time"] < timesteps - 1])
            explored = {cell: record["count"] for cell, record in policy.explored.cells.items()}
            self.assertEqual(explored, expected)
            # a cell is credited to the robot in its own map when no other robot observed it earlier:
            # on a same-timestep tie, the own observation arrives before the other's broadcast
            for cell in explored:
                at_cell = [o for o in observations if (o["x"], o["y"]) == cell]
                own = min((o["time"] for o in at_cell if o["robot"] == name), default=None)
                other = min((o["time"] for o in at_cell if o["robot"] != name), default=None)
                credited = own is not None and (other is None or own <= other)
                self.assertEqual(policy.explored.first(*cell)[0] == name, credited)

    def test_without_rounds_nothing_is_sent_and_the_map_is_only_own(self):
        policies, _, medium, results = run_market(communication_rounds=0)
        self.assertEqual(medium.traffic, [])
        observations = [observation for step in results["observations"] for observation in step]
        for name, policy in policies.items():
            explored = {cell: record["count"] for cell, record in policy.explored.cells.items()}
            self.assertEqual(explored, cell_counts([o for o in observations if o["robot"] == name]))


class TestMarketTraffic(unittest.TestCase):
    def test_market_messages_are_counted_by_type(self):
        timesteps = 60
        _, _, medium, _ = run_market(communication_rounds=3, timesteps=timesteps)
        summary = traffic_summary(medium.traffic, timesteps)
        per_type = {message_type: totals["messages"] for message_type, totals in summary["per-type"].items()}
        # two offers broadcast, both bid by both contractors, both awarded to con-1, the first one completed
        self.assertEqual(per_type, {"location": 3 * (timesteps - 1), "ep-offer": 2, "ep-bid": 4,
                                    "ep-award": 2, "ep-completed": 1})
        self.assertEqual(summary["per-type"]["ep-offer"]["bytes-delivered"],
                         2 * summary["per-type"]["ep-offer"]["bytes-transmitted"])
        self.assertEqual(summary["bytes-transmitted"], sum(record["bytes"] for record in medium.traffic))
        self.assertEqual(sum(summary["bytes-per-timestep"]), summary["bytes-transmitted"])


def idle_contractor(name, x, y, budget=200, timestep=0):
    """A contractor without commitments at (x, y), at the given timestep"""
    contractor = MAREX_Contractor({"policy-name": name, "seed": 1, "budget": budget}, None)
    Robot(name, x, y, 0).assign_policy(contractor)
    contractor.timestep = timestep
    return contractor


def offer_of(ep, offer_id="pioneer-0", prize=10):
    return EPOffer(offer_id, ep, "pioneer", prize)


class TestMarginalCostBids(unittest.TestCase):
    EP = ExplorationPackage(10, 14, 10, 14, step=2)

    def test_idle_contractor_bids_travel_plus_coverage(self):
        contractor = idle_contractor("con-1", 0, 0)
        canbid, bid = contractor.can_bid(offer_of(self.EP))
        self.assertTrue(canbid)
        # the schedule starts at the next timestep, at the current position
        self.assertEqual(bid, contractor.schedule_end([self.EP], [0, 0], 1) - 1)
        self.assertGreater(bid, 0)

    def test_nearer_idle_contractor_bids_lower_and_wins(self):
        near, far = idle_contractor("con-2", 8, 8), idle_contractor("con-1", 0, 0)
        owner = EPAgent("pioneer")
        offer = owner.offer(self.EP, prize=10)
        for contractor in [near, far]:
            _, bid = contractor.can_bid(offer_of(self.EP, offer.offer_id))
            owner.receive_bid(offer.offer_id, contractor.name, bid)
        self.assertLess(offer.bids["con-2"], offer.bids["con-1"])
        self.assertEqual(owner.clear(offer.offer_id).assigned_to_name, "con-2")

    def test_bid_starts_where_the_current_ep_ends(self):
        contractor = idle_contractor("con-1", 0, 0)
        # inside an EP whose last step is next to the offered EP
        current = ExplorationPackage(0, 8, 0, 8, step=2)
        contractor.plan = [{"x": 0, "y": 0, "t": 1, "ep": current}, {"x": 9, "y": 9, "t": 2, "ep": current},
                           {"x": 50, "y": 50, "t": 3, "ep": None}]
        x, y, t, ep = contractor.schedule_start()
        self.assertEqual((x, y, t, ep), (9, 9, 3, current))
        _, bid = contractor.can_bid(offer_of(self.EP))
        _, bid_from_origin = idle_contractor("con-2", 0, 0).can_bid(offer_of(self.EP))
        self.assertLess(bid, bid_from_origin)

    def test_outstanding_bids_count_as_commitments(self):
        contractor = idle_contractor("con-1", 0, 0)
        other = offer_of(ExplorationPackage(30, 34, 30, 34, step=2), "pioneer-1")
        _, alone = contractor.can_bid(offer_of(self.EP))
        _, first = contractor.can_bid(other)
        contractor.epagent.bid(other)
        canbid, with_other = contractor.can_bid(offer_of(self.EP))
        self.assertTrue(canbid)
        # with the other EP held, the bid is the change of a longer schedule, not the cost of the EP alone
        self.assertEqual(with_other, contractor.schedule_end([other.ep, self.EP], [0, 0], 1)
                         - contractor.schedule_end([other.ep], [0, 0], 1))
        self.assertNotEqual(with_other, alone)

    def test_infeasible_offer_gets_no_bid(self):
        contractor = idle_contractor("con-1", 0, 0, budget=20)
        self.assertEqual(contractor.can_bid(offer_of(self.EP)), (False, None))


if __name__ == "__main__":
    unittest.main()
