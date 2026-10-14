"""
epmarket.py

Classes of the MultiResolutionMultiRobot paper that implement a market for exploration packages.

The market has no shared state: every agent keeps its own ledger (EPAgent), and agents interact only
by exchanging messages through the communication medium (see marex_policies.py and
DESIGN-COMMUNICATION.md). Offers are identified by their offer id, because every recipient of a
message works with its own copy of the offer.

The currency of the market is one timestep of exploration budget. An offer carries the offering agent's
estimate of the value of the EP (its prize), which does not restrict the bids. A contractor bids the
marginal cost of the EP, the number of timesteps by which it lengthens its schedule, and the offer is
awarded to the lowest bid, at the price of that bid.

"""
import pprint
import textwrap

# the communication rounds between the broadcast of an offer and its award: the bids are sent in the round
# after the offer, and the offering agent clears the offer in the round after that. A bid that was not
# awarded by then is lost (the award goes only to the winner).
AWARD_DELAY_ROUNDS = 2

class EPOffer:
    """An offer for the execution of an exploration package"""
    def __init__(self, offer_id, ep, offering_agent_name, prize):
        self.offer_id = offer_id
        self.ep = ep
        self.offering_agent_name = offering_agent_name
        self.prize = prize
        self.bid_prize = prize
        self.bids = {} # bidder name -> value, kept by the offering agent
        self.assigned_to_name = None # name of the assigned agent
        self.executed = False
        self.real_value = 0
        self.sent_round = None # the offering agent's round count when the offer was sent
        self.bid_round = None # the bidder's round count when it bid on the offer

    def __repr__(self):
        pretty_str = "EPOffer: " +  pprint.pformat(self.__dict__, indent=4)
        return pretty_str


class EPAgent:
    """The robot-local ledger of an agent participating in the market"""
    def __init__(self, name):
        self.name = name
        self.money = 0
        self.offer_count = 0 # used to create globally unique offer ids
        self.commitments = {} # offer id -> offer, the deals we won and have to execute
        self.outstanding_offers = {} # offer id -> offer, our offers waiting for clearing
        self.outstanding_bids = {} # offer id -> offer, the offers we bid on and were not awarded
        self.agreed_deals = {} # offer id -> offer, our offers awarded to a contractor
        self.terminated_deals = []
        self.declined_offers = []
        self.lost_bids = [] # the offers we bid on and were not awarded

    def __repr__(self):
        retval = f"Agent: {self.name}\n"
        retval+= textwrap.indent("Commitments: " + pprint.pformat(self.commitments), " " * 4) + "\n"
        retval+= textwrap.indent("Outstanding offers: " + pprint.pformat(self.outstanding_offers), " " * 4) + "\n"
        retval+= textwrap.indent("Outstanding bids: " + pprint.pformat(self.outstanding_bids), " " * 4) + "\n"
        retval+= textwrap.indent("Agreed deals: " + pprint.pformat(self.agreed_deals), " " * 4) + "\n"
        retval+= textwrap.indent("Terminated deals: " + pprint.pformat(self.terminated_deals), " " * 4) + "\n"
        return retval

    #
    # The offering agent's side
    #
    def offer(self, ep, prize):
        """Create a new offer with a globally unique id, waiting for bids"""
        epoff = EPOffer(f"{self.name}-{self.offer_count}", ep, self.name, prize)
        self.offer_count += 1
        self.outstanding_offers[epoff.offer_id] = epoff
        return epoff

    def receive_bid(self, offer_id, bidder_name, value):
        """Record a bid received for one of our offers"""
        self.outstanding_offers[offer_id].bids[bidder_name] = value

    def clear(self, offer_id):
        """Award our offer to the lowest bidder, ties broken by name order. Returns the
        awarded offer, or None if the offer received no bids and was declined."""
        epoff = self.outstanding_offers.pop(offer_id)
        if not epoff.bids:
            self.declined_offers.append(epoff)
            return None
        winner = min(sorted(epoff.bids), key=lambda name: epoff.bids[name])
        epoff.assigned_to_name = winner
        epoff.bid_prize = epoff.bids[winner]
        self.agreed_deals[offer_id] = epoff
        return epoff

    def offer_finished(self, offer_id, real_value):
        """The contractor executed our offer: pay the prize, receive the real value."""
        epoff = self.agreed_deals.pop(offer_id)
        epoff.real_value = real_value
        epoff.executed = True
        self.terminated_deals.append(epoff)
        self.money += epoff.real_value - epoff.bid_prize

    #
    # The contractor's side
    #
    def bid(self, epoff, round=None):
        """Record that we bid on an offer (our own copy of it), in the given communication round"""
        epoff.bid_round = round
        self.outstanding_bids[epoff.offer_id] = epoff

    def expire_bids(self, round):
        """Forget the bids that were not awarded within AWARD_DELAY_ROUNDS rounds: they were lost"""
        for offer_id, epoff in list(self.outstanding_bids.items()):
            if epoff.bid_round is not None and round - epoff.bid_round > AWARD_DELAY_ROUNDS:
                self.lost_bids.append(self.outstanding_bids.pop(offer_id))

    def won(self, offer_id, price):
        """We were awarded the offer at the given price; it becomes a commitment"""
        epoff = self.outstanding_bids.pop(offer_id)
        epoff.bid_prize = price
        epoff.assigned_to_name = self.name
        self.commitments[offer_id] = epoff
        return epoff

    def commitment_executed(self, offer_id, real_value):
        """We executed a commitment: we receive the prize"""
        epoff = self.commitments.pop(offer_id)
        epoff.real_value = real_value
        epoff.executed = True
        self.terminated_deals.append(epoff)
        self.money += epoff.bid_prize
        return epoff
