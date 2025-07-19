"""
epmarket.py

Classes of the MultiResolutionMultiRobot paper implement a market for exploration packages.

"""
import pprint
import textwrap

class EPOffer:
    """An offer for the execution of an exploration package"""
    def __init__(self, ep, offering_agent_name, prize):
        self.ep = ep
        self.offering_agent_name = offering_agent_name
        self.prize = prize
        self.bid_prize = prize
        self.bids = {} 
        self.assigned_to_name = None # name of the assigned agent
        self.executed = False
        self.real_value = 0

    def __repr__(self):
        pretty_str = "EPOffer: " +  pprint.pformat(self.__dict__, indent=4)
        #print(pretty_str)
        return pretty_str
    
class EPAgent: 
    """An agent participating in the market"""
    def __init__(self, name):
        self.name = name
        self.epm = None
        self.money = 0
        self.commitments = [] # list of the offers to which we committed
        self.outstanding_offers = {}
        self.outstanding_bids = {}
        self.agreed_deals = []
        self.terminated_deals = []
        self.policy = None
        

    def __repr__(self):
        retval = f"Agent: {self.name}\n"
        retval+= textwrap.indent("Commitments: " + pprint.pformat(self.commitments), " " * 4) + "\n"
        retval+= textwrap.indent("Outstanding offers: " + pprint.pformat(self.outstanding_offers), " " * 4) + "\n"
        retval+= textwrap.indent("Outstanding bids: " + pprint.pformat(self.outstanding_bids), " " * 4) + "\n"
        retval+= textwrap.indent("Agreed deals: " + pprint.pformat(self.agreed_deals), " " * 4) + "\n"
        retval+= textwrap.indent("Terminated deals: " + pprint.pformat(self.terminated_deals), " " * 4) + "\n"
        return retval

    def join(self, epm):
        """Join the epm"""
        self.epm = epm
        epm.agents[self.name] = self

    def offer(self, ep, prize):
        """Create an offer and add it to the epm"""
        epoff = EPOffer(ep, self.name, prize)
        self.epm.add_offer(epoff)
        self.outstanding_offers[epoff] = epoff
        return epoff

    def bid(self, epoff, value):
        """Called by the agent to indicate that it made an offer"""
        epoff.bids[self.name] = value        
        self.outstanding_bids[epoff] = epoff

    def won(self, epoff):
        """Called by the epoff during clearing to show that 
        you now have a commitment"""
        self.commitments.append(epoff)
        if self.policy is not None:
            self.policy.won(epoff)

    def commitment_executed(self, epoff, real_value):
        """Called by the agent to indicate that the commitment was executed"""
        epoff.real_value = real_value
        epoff.executed = True
        self.money += epoff.bid_prize
        # FIXME: this is probably happening after the last ep
        if epoff in self.commitments:
            self.commitments.remove(epoff)
            offering_agent = self.epm.agents[epoff.offering_agent_name]
            offering_agent.offer_finished(epoff)
            print(f"commitment_executed called for epoff {epoff}")
        else:
            print(f"For some reason, commitment_executed called although epoff {epoff} is not in commitments")
            

    def offer_accepted(self, epoff):
        """The offer was accepted"""
        self.agreed_deals.append(epoff)
        self.outstanding_offers.pop(epoff)

    def offer_declined(self, epoff):
        """The offer was not accepted. In a more sophisticated 
        system there should be some way to repeat the bid..."""
        self.outstanding_offers.pop(epoff)

    def offer_finished(self, epoff):
        """Called by the epmarket to indicated that the offer was finished. Pay the prize, receive the real value."""
        self.agreed_deals.remove(epoff)
        self.terminated_deals.append(epoff)
        self.money += epoff.real_value - epoff.bid_prize

class EPMarket:
    """A market for ExplorationPackages. The idea is that the agent is offering the exploration package and a prize money for the exploration. The offering agent will keep the value found through the execution of the exploration."""

    def __init__(self):
        self.agents = {}   
        self.pending_offers = [] 
        self.accepted_offers = [] 
        # dictionary allowing to match the eps to offers
        self.ep_to_offer = {}

    def __repr__(self):
        retval = "EPMarket:\n"
        retval+= textwrap.indent("Pending offers: " + pprint.pformat(self.pending_offers), " " * 4) + "\n"
        retval+= textwrap.indent("Accepted offers: " + pprint.pformat(self.accepted_offers), " " * 4) + "\n"
        # "Agents:    " + pprint.pformat(self.agents, indent=4)
        return retval    

    def join(self, agent: EPAgent):
        self.agents[agent.name] = agent
        agent.epm = self

    def add_offer(self, epoff):
        self.pending_offers.append(epoff)
        self.ep_to_offer[epoff.ep] = epoff

    def clearing(self):
        """Award the the offers to the highest bidder, or reject it."""
        for epoff in self.pending_offers:
            minbid = float('inf')
            offering_agent = self.agents[epoff.offering_agent_name]
            bestbidder = None
            for agentname in epoff.bids:
                if epoff.bids[agentname] < minbid:
                    minbid = epoff.bids[agentname]
                    bestbidder = agentname
            if bestbidder is None:
                offering_agent.offer_declined(epoff)
            else:
                offering_agent.offer_accepted(epoff)
                bestbidderagent = self.agents[bestbidder]
                bestbidderagent.won(epoff)
                self.accepted_offers.append(epoff)
        self.pending_offers = []

class EPM:
    """A singleton class for a single market. Simplifies the implementation"""
    _instance = None  # Class-level attribute to store the instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EPM, cls).__new__(cls)
            cls._instance.epm = EPMarket()
        return cls._instance
