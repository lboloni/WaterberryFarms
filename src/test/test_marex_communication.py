import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium, traffic_summary
from papers.y2027_marex.epmarket import AWARD_DELAY_ROUNDS, EPAgent, EPOffer
from papers.y2027_marex.exploration_package import ExplorationPackage
from papers.y2027_marex.marex_policies import MAREX_DEFAULTS, MAREX_Contractor, MAREX_Pioneer
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
        # every package is completed when con-1's plan leaves it, the last one included; the price is the
        # winning bid, in timesteps of exploration budget, not the prize of the offer
        self.assertEqual([deal.offer_id for deal in pioneer.terminated_deals], ["pioneer-0", "pioneer-1"])
        prices = [deal.bid_prize for deal in pioneer.terminated_deals]
        self.assertNotIn(10, prices)
        self.assertEqual(contractor.money, sum(prices))
        self.assertEqual(pioneer.money, 0 - sum(prices))

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
        # two offers broadcast, both bid by both contractors, both awarded to con-1 and completed
        self.assertEqual(per_type, {"location": 3 * (timesteps - 1), "ep-offer": 2, "ep-bid": 4,
                                    "ep-award": 2, "ep-completed": 2})
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


class TestLostBids(unittest.TestCase):
    EP = ExplorationPackage(10, 14, 10, 14, step=2)

    def test_lost_bid_expires(self):
        contractor = idle_contractor("con-1", 0, 0)
        lost = offer_of(ExplorationPackage(30, 34, 30, 34, step=2), "pioneer-1")
        contractor.epagent.bid(lost, round=5)
        _, idle_bid = idle_contractor("con-2", 0, 0).can_bid(offer_of(self.EP))
        # while the auction may still be awarded to us, the bid counts as a commitment
        contractor.epagent.expire_bids(5 + AWARD_DELAY_ROUNDS)
        self.assertIn("pioneer-1", contractor.epagent.outstanding_bids)
        self.assertNotEqual(contractor.can_bid(offer_of(self.EP))[1], idle_bid)
        # one round later it was lost: forgotten, and the bid is that of an idle contractor
        contractor.epagent.expire_bids(5 + AWARD_DELAY_ROUNDS + 1)
        self.assertEqual(contractor.epagent.outstanding_bids, {})
        self.assertEqual([offer.offer_id for offer in contractor.epagent.lost_bids], ["pioneer-1"])
        self.assertEqual(contractor.can_bid(offer_of(self.EP))[1], idle_bid)

    def test_loser_forgets_the_lost_auction(self):
        # both contractors bid the same on the offer at timestep 1; con-2 loses by name order
        policies, commitments, _, _ = run_market(communication_rounds=3)
        loser = policies["con-2"].epagent
        self.assertEqual(loser.commitments, {})
        self.assertEqual(loser.outstanding_bids, {})
        self.assertEqual(sorted(offer.offer_id for offer in loser.lost_bids), ["pioneer-0", "pioneer-1"])


class TestCompletionReport(unittest.TestCase):
    EP = ExplorationPackage(10, 14, 10, 14, step=2)
    OTHER = ExplorationPackage(30, 34, 30, 34, step=2)

    def exploring_contractor(self):
        """A contractor at (10, 10) exploring its only EP, whose plan leaves the EP after two steps"""
        contractor = idle_contractor("con-1", 10, 10)
        epoff = offer_of(self.EP)
        epoff.bid_prize = 5
        contractor.epagent.commitments[epoff.offer_id] = epoff
        contractor.current_epoffer = epoff
        contractor.plan = [{"x": 10, "y": 10, "t": 0, "ep": self.EP}, {"x": 11, "y": 10, "t": 1, "ep": self.EP},
                           {"x": 12, "y": 10, "t": 2, "ep": None}]
        return contractor

    def test_last_ep_reported_when_the_plan_leaves_it(self):
        contractor = self.exploring_contractor()
        contractor.report_left_ep()
        self.assertEqual(contractor.outbox, [])
        contractor.plan.pop(0)
        contractor.report_left_ep()
        self.assertEqual(contractor.outbox, [])
        # the next step is a random waypoint (ep None): the EP is finished
        contractor.plan.pop(0)
        contractor.report_left_ep()
        self.assertEqual(contractor.outbox, [("pioneer", {"type": "ep-completed", "offer-id": "pioneer-0"})])
        self.assertIsNone(contractor.current_epoffer)
        self.assertEqual(contractor.epagent.commitments, {})

    def test_finished_ep_does_not_count_in_the_bids(self):
        contractor = self.exploring_contractor()
        contractor.plan = contractor.plan[2:]
        _, idle_bid = idle_contractor("con-2", 10, 10).can_bid(offer_of(self.OTHER, "pioneer-1"))
        # before the report, the finished EP would lengthen the schedule
        self.assertNotEqual(contractor.can_bid(offer_of(self.OTHER, "pioneer-1"))[1], idle_bid)
        contractor.report_left_ep()
        self.assertEqual(contractor.can_bid(offer_of(self.OTHER, "pioneer-1"))[1], idle_bid)


def destroyed(x, y, t=0):
    """An observation of a destroyed plant (a detection)"""
    return {"x": x, "y": y, "time": t, "TYLCV": {"value": 0.0}}


def infected(x, y, t=0):
    return {"x": x, "y": y, "time": t, "TYLCV": {"value": 0.5}}


def healthy(x, y, t=0):
    return {"x": x, "y": y, "time": t, "TYLCV": {"value": 1.0}}


def offering_contractor(name="con-1", option=True, budget=200, **parameters):
    """An idle contractor at (0, 0) on a 31x31 field, with the contractors-offer option and the given
    parameters of MAREX (MAREX_DEFAULTS)"""
    contractor = MAREX_Contractor({"policy-name": name, "seed": 1, "budget": budget,
                                   "contractors-offer": option, **parameters}, None)
    robot = Robot(name, 0, 0, 0)
    robot.im = SizedEstimator()
    robot.assign_policy(contractor)
    contractor.timestep = 0
    return contractor


def observe(contractor, *observations):
    """Feed observations one by one, as act does after every move"""
    for obs in observations:
        contractor.observations.append(obs)
        contractor.offer_own_streaks()


class TestContractorOffers(unittest.TestCase):
    STREAK = [destroyed(15, 15), destroyed(16, 15), healthy(17, 15)]

    def test_contractor_offers_a_streak_outside_its_eps(self):
        contractor = offering_contractor()
        observe(contractor, *self.STREAK)
        [offer] = contractor.epagent.outstanding_offers.values()
        self.assertEqual((offer.ep.x_min, offer.ep.x_max, offer.ep.y_min, offer.ep.y_max), (12, 19, 12, 18))
        self.assertIn(offer.ep, contractor.known_eps)

    def test_contractor_does_not_offer_inside_an_ep(self):
        contractor = offering_contractor()
        committed = offer_of(ExplorationPackage(10, 20, 10, 20, step=2), "pioneer-0")
        contractor.epagent.commitments[committed.offer_id] = committed
        observe(contractor, *self.STREAK)
        self.assertEqual(contractor.epagent.outstanding_offers, {})
        self.assertEqual(contractor.detections, [])

    def test_no_offer_overlapping_a_known_ep(self):
        contractor = offering_contractor()
        contractor.known_eps.append(ExplorationPackage(14, 24, 14, 24, step=2))
        observe(contractor, *self.STREAK)
        self.assertEqual(contractor.epagent.outstanding_offers, {})

    def test_option_off_contractors_never_offer(self):
        contractor = offering_contractor(option=False)
        observe(contractor, *self.STREAK)
        self.assertEqual(contractor.epagent.outstanding_offers, {})

    def test_offering_contractor_can_win_its_own_offer(self):
        contractor = offering_contractor()
        observe(contractor, *self.STREAK)
        [offer] = contractor.epagent.outstanding_offers.values()
        # the offer is broadcast, and we bid on it without a message
        contractor.rounds = 1
        contractor.before_send()
        self.assertEqual([content["type"] for _, content in contractor.outbox], ["ep-offer"])
        self.assertIn("con-1", offer.bids)
        # cleared without other bids: we win it, and no award message is sent
        contractor.outbox = []
        contractor.rounds = 1 + AWARD_DELAY_ROUNDS
        contractor.replan_needed = False
        contractor.before_send()
        self.assertEqual(contractor.outbox, [])
        self.assertIn(offer.offer_id, contractor.epagent.commitments)
        self.assertTrue(contractor.replan_needed)
        # its completion is settled in our own ledger, without a message
        contractor.report_completion(offer)
        self.assertEqual(contractor.outbox, [])
        self.assertEqual(contractor.epagent.commitments, {})
        self.assertEqual([deal.offer_id for deal in contractor.epagent.terminated_deals], [offer.offer_id] * 2)
        self.assertEqual(contractor.epagent.money, 0)


class TestMarexParameters(unittest.TestCase):
    STREAK = [destroyed(15, 15), destroyed(16, 15), healthy(17, 15)]

    def test_defaults(self):
        contractor = offering_contractor()
        for key, value in MAREX_DEFAULTS.items():
            self.assertEqual(getattr(contractor, key.replace("-", "_")), value)
        self.assertEqual(contractor.epagent.award_delay_rounds, MAREX_DEFAULTS["award-delay-rounds"])

    def test_ep_step_border_and_prize_factor(self):
        contractor = offering_contractor(**{"ep-step": 1, "ep-border": 5, "prize-factor": 1.0})
        observe(contractor, *self.STREAK)
        [offer] = contractor.epagent.outstanding_offers.values()
        self.assertEqual((offer.ep.x_min, offer.ep.x_max, offer.ep.y_min, offer.ep.y_max), (10, 21, 10, 20))
        self.assertEqual(offer.ep.step, 1)
        self.assertEqual(offer.prize, 11 * 10)

    def test_streak_below_destroyed_only(self):
        # with 0.5, an infected plant (0.5) ends the streak, like a healthy one
        contractor = offering_contractor(**{"streak-below": 0.5})
        observe(contractor, destroyed(15, 15), infected(16, 15))
        self.assertEqual(len(contractor.epagent.outstanding_offers), 1)
        observe(contractor, infected(20, 25), healthy(21, 25))
        self.assertEqual(len(contractor.epagent.outstanding_offers), 1)

    def test_streak_below_infected_and_destroyed(self):
        # with 1.0 (the default), the infected and the destroyed plants make up one streak, which a healthy plant ends
        contractor = offering_contractor(**{"streak-below": 1.0})
        observe(contractor, destroyed(15, 15), infected(16, 15))
        self.assertEqual(contractor.epagent.outstanding_offers, {})
        self.assertEqual(contractor.detections, [[15, 15], [16, 15]])
        observe(contractor, healthy(17, 15))
        self.assertEqual(len(contractor.epagent.outstanding_offers), 1)

    def test_award_delay_rounds(self):
        contractor = offering_contractor(**{"award-delay-rounds": 4})
        lost = offer_of(ExplorationPackage(30, 34, 30, 34, step=2), "pioneer-1")
        contractor.epagent.bid(lost, round=5)
        contractor.epagent.expire_bids(5 + 4)
        self.assertIn("pioneer-1", contractor.epagent.outstanding_bids)
        contractor.epagent.expire_bids(5 + 4 + 1)
        self.assertEqual(contractor.epagent.outstanding_bids, {})


if __name__ == "__main__":
    unittest.main()
