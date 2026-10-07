import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import PerfectCommunicationMedium
from papers.y2027_mrmr.epmarket import EPAgent
from papers.y2027_mrmr.exploration_package import ExplorationPackage
from papers.y2027_mrmr.mrmr_policies import MRMR_Contractor, MRMR_Pioneer
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
    Returns the policies, the medium, and the contractors' commitments at the end of every timestep."""
    environment = HealthyEnvironment()
    medium = PerfectCommunicationMedium(environment)
    policies = {"pio": MRMR_Pioneer({"policy-name": "pioneer", "seed": 1, "budget": timesteps}, None),
                "con-1": MRMR_Contractor({"policy-name": "contractor-1", "seed": 2, "budget": timesteps}, None),
                "con-2": MRMR_Contractor({"policy-name": "contractor-2", "seed": 3, "budget": timesteps}, None)}
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

    simulate_1day(environment=environment, robots=robots, estimator=SizedEstimator(),
                  evaluator=ZeroEvaluator(), timesteps=timesteps, estimator_interval=10,
                  communication=medium, communication_rounds=communication_rounds,
                  after_timestep=after_timestep)
    return policies, commitments


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
        policies, commitments = run_market(communication_rounds=3)
        # offered after timestep 0, broadcast, bid and awarded in the rounds of timestep 1
        self.assertEqual(commitments[0], {"con-1": [], "con-2": []})
        # both contractors bid the prize; the tie goes to con-1, first in name order
        self.assertEqual(commitments[1], {"con-1": ["pioneer-0", "pioneer-1"], "con-2": []})

    def test_completion_settles_the_money_on_both_sides(self):
        policies, commitments = run_market(communication_rounds=3)
        pioneer, contractor = policies["pio"].epagent, policies["con-1"].epagent
        # the first package is completed when con-1 moves on to the second one
        self.assertEqual([deal.offer_id for deal in pioneer.terminated_deals], ["pioneer-0"])
        self.assertEqual(contractor.money, 10)
        self.assertEqual(pioneer.money, 0 - 10)

    def test_auction_spreads_over_timesteps_with_one_round(self):
        policies, commitments = run_market(communication_rounds=1)
        # offer in timestep 1, bids in timestep 2, award in timestep 3
        self.assertEqual(commitments[2], {"con-1": [], "con-2": []})
        self.assertEqual(commitments[3], {"con-1": ["pioneer-0", "pioneer-1"], "con-2": []})

    def test_robots_hold_no_references_to_each_other(self):
        policies, _ = run_market(communication_rounds=3)
        others = {id(policy) for policy in policies.values()} | {id(policy.epagent) for policy in policies.values()}
        for policy in policies.values():
            own = {id(policy), id(policy.epagent)}
            for value in list(vars(policy).values()) + list(vars(policy.epagent).values()):
                self.assertNotIn(id(value), others - own)


if __name__ == "__main__":
    unittest.main()
