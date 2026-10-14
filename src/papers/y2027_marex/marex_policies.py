"""
marex_policies.py

Implementation of the various MAREX policies

"""

import numpy as np
import copy

from policy import Policy, AbstractCommunicateAndFollowPath
from communication import Message
from information_model import ObservationRecord
#from papers.y2027_marex.epmarket import EPAgent, EPOffer
#from papers.y2027_marex.exploration_package import ExplorationPackage, ExplorationPackageSet
#from papers.y2027_marex.xyplans import create_random_waypoints, xyplan_from_waypoints, xyplan_from_ep_path
from .epmarket import AWARD_DELAY_ROUNDS, EPAgent, EPOffer
from .exploration_package import ExplorationPackage, ExplorationPackageSet
from .xyplans import create_random_waypoints, xyplan_from_waypoints, xyplan_from_ep_path


def ep_to_content(ep):
    """The exploration package as plain data, for sending in a message"""
    return {"x_min": ep.x_min, "x_max": ep.x_max, "y_min": ep.y_min, "y_max": ep.y_max, "step": ep.step}


def ep_from_content(content):
    """Reconstruct an exploration package from a message"""
    return ExplorationPackage(content["x_min"], content["x_max"], content["y_min"], content["y_max"], content["step"])


class MAREX_Policy(Policy):
    """Implements the common ancestor of the MAREX (multiresolution multirobot models). The agents 
    interact only through messages: ep-offer, ep-bid, ep-award, ep-completed and location
    (DESIGN-COMMUNICATION.md). Every agent broadcasts its location at every timestep, and builds
    from the broadcasts a map of the explored cells."""

    def __init__(self, exp_policy, exp_env):
        self.exp_policy = exp_policy
        self.exp_env = exp_env
        self.name = exp_policy["policy-name"]
        self.timestep = -1 # local way to keep track of time, based on act calls
        # the robot-local ledger of the market
        self.epagent = EPAgent(self.name)
        self.outbox = [] # (destination, content) pairs to send in the next communication round
        self.rounds = 0 # the communication rounds seen so far, counted across timesteps
        self.observations = []
        self._explored = None # the robot-local map of the explored cells, created at first use
        # initializing the agent's random generator
        seed = self.exp_policy["seed"]
        self.random = np.random.default_rng(seed)
        # whether the contractors also create EPs from their own streaks (DESIGN-Market.md); with the option
        # off, only the pioneer creates EPs, and it checks the overlap only against its own awarded deals
        self.contractors_offer = bool(self.exp_policy.get("contractors-offer", False))
        self.detections = [] # the streak of detections in progress
        self.known_eps = [] # the EPs of all the offers we heard or made

    def create_randxy(self, xcurrent=0, ycurrent=0, t=0):
        """Create a random xy plan starting from the specified position and time t. We assume that the robot is connected at this point"""
        geom_x_min = 0
        geom_x_max = self.robot.im.width - 1
        geom_y_min = 0
        geom_y_max = self.robot.im.height - 1
        budget = self.exp_policy["budget"] - t

        randwp = create_random_waypoints(self.random, xcurrent, ycurrent, geom_x_min, geom_x_max, geom_y_min, geom_y_max, budget)
        randxy = xyplan_from_waypoints(randwp, t, vel=1, ep=None)        
        return randxy[0:int(budget)]

    @property
    def explored(self):
        """The map of the explored cells: our own observations and the locations broadcast by the
        other agents, which arrive one timestep later"""
        if self._explored is None:
            self._explored = ObservationRecord(self.robot.im.width, self.robot.im.height)
        return self._explored

    def add_observation(self, obs):
        """MAREX policies collect the observations"""
        obs["name"] = self.name
        self.observations.append(obs)
        self.explored.add(obs)

    def act_send(self, round):
        """Send the queued messages. In round 0, broadcast the location of the latest observation,
        which is the current location of the robot."""
        self.rounds += 1
        if round == 0 and self.observations:
            obs = self.observations[-1]
            self.outbox.append((None, {"type": "location", "x": obs["x"], "y": obs["y"], "time": obs["time"]}))
        self.before_send()
        for destination, content in self.outbox:
            self.robot.com.send(self.robot, destination, Message(content))
        self.outbox = []

    def act_receive(self, round, messages):
        """Dispatch the received messages on their type"""
        for message in messages:
            getattr(self, "on_" + message.content["type"].replace("-", "_"))(message)

    def on_location(self, message):
        """Another agent explored a cell"""
        content = message.content
        self.explored.add({"x": content["x"], "y": content["y"], "time": content["time"],
                           "robot": message.sender_name})

    #
    # streaks and offers
    #
    def track_streak(self, obs):
        """Extend the streak of detections with an observation; returns the streak if the observation ended
        it, otherwise None. A streak is a run of consecutive observations of destroyed plants."""
        if obs["TYLCV"]["value"] == 0.0:
            self.detections.append([obs["x"], obs["y"]])
            return None
        streak, self.detections = self.detections, []
        return streak if streak else None

    def build_offer(self, streak):
        """The EP around a streak and its prize, or None if it overlaps an EP that rules it out: with the
        contractors offering, any EP we heard or made an offer for; otherwise one of our awarded deals"""
        streak = np.array(streak)
        im = self.robot.im
        geom_x_min, geom_x_max = 0, im.width - 1
        geom_y_min, geom_y_max = 0, im.height - 1
        # the bounding box of the streak, extended by a border and clipped to the field
        # border = 3 * streak.shape[0] would make the EP grow with the streak
        border = 3
        x_min = max(geom_x_min, np.min(streak[:, 0]) - border)
        x_max = min(geom_x_max, np.max(streak[:, 0]) + border)
        y_min = max(geom_y_min, np.min(streak[:, 1]) - border)
        y_max = min(geom_y_max, np.max(streak[:, 1]) + border)
        ep = ExplorationPackage(x_min, x_max, y_min, y_max, step = 2)
        # the prize is proportional to the area
        prize = (x_max - x_min) * (y_max - y_min) * 0.1
        if self.contractors_offer:
            ruling = self.known_eps
        else:
            ruling = [epold.ep for epold in self.epagent.agreed_deals.values()]
        if any(other.overlap(ep) for other in ruling):
            return None
        return ep, prize

    def make_offer(self, ep, prize):
        """Create an offer, broadcast in the next communication round"""
        self.known_eps.append(ep)
        return self.epagent.offer(ep, prize)

    def before_send(self):
        """Clear the offers whose bids arrived (AWARD_DELAY_ROUNDS rounds after sending them), then broadcast
        the new offers"""
        for epoff in list(self.epagent.outstanding_offers.values()):
            if epoff.sent_round is not None and self.rounds >= epoff.sent_round + AWARD_DELAY_ROUNDS:
                awarded = self.epagent.clear(epoff.offer_id)
                if awarded is None:
                    continue
                if awarded.assigned_to_name == self.robot.name:
                    self.on_self_award(awarded)
                else:
                    self.outbox.append((awarded.assigned_to_name, {"type": "ep-award",
                        "offer-id": awarded.offer_id, "price": awarded.bid_prize}))
        for epoff in self.epagent.outstanding_offers.values():
            if epoff.sent_round is None:
                epoff.sent_round = self.rounds
                self.outbox.append((None, {"type": "ep-offer", "offer-id": epoff.offer_id,
                    "ep": ep_to_content(epoff.ep), "prize": epoff.prize}))
                self.on_own_offer_sent(epoff)

    def on_own_offer_sent(self, epoff):
        """Hook: our offer was just broadcast (a contractor bids on its own offers here)"""
        pass

    def on_self_award(self, epoff):
        """Hook: we won our own offer"""
        pass

    def on_ep_offer(self, message):
        """Remember the EP of every offer we hear, for the overlap check"""
        self.known_eps.append(ep_from_content(message.content["ep"]))

    def on_ep_bid(self, message):
        self.epagent.receive_bid(message.content["offer-id"], message.sender_name, message.content["value"])

    def on_ep_completed(self, message):
        self.epagent.offer_finished(message.content["offer-id"], message.content["real-value"])

class MAREX_Pioneer(MAREX_Policy):
    """Implements the Pioneer agent for the MAREX paper: it explores with random waypoints, offers EPs around
    its streaks of detections, and never explores EPs itself"""

    def __init__(self, exp_policy, exp_env):
        super().__init__(exp_policy, exp_env)
        self.plan = []
        self.replan_needed = True

    def replan(self):
        if not self.replan_needed and self.plan != []:
            return    
        self.plan = []    
        randxy = self.create_randxy(xcurrent=self.robot.x, ycurrent=self.robot.y, t=self.timestep)
        self.plan += randxy
        self.replan_needed = False        

    def act(self, delta_t):
        """The primary behavior of the agent. """
        self.timestep += delta_t
        self.replan()
        # assert self.plan[0]["t"] == self.timestep
        if len(self.plan) == 0:
            print("No plan???")
        if self.plan[0]["t"] != self.timestep:
            print("assertion broken here")

        self.robot.add_action(f"loc [{self.plan[0]['x']}, {self.plan[0]['y']}]")
        self.plan.pop(0) # move on with the plan
        # the streaks of detections, around which we offer EPs; FIXME: only the destroyed plants count
        if len(self.observations) == 0:
            return
        streak = self.track_streak(self.observations[-1])
        if streak is None:
            return
        offer = self.build_offer(streak)
        if offer is not None:
            self.make_offer(*offer)


class MAREX_Contractor(MAREX_Policy):
    """Implements the Contractor agent for the MAREX paper"""
    # the bounds of the EP path search, as counts of evaluated combinations: they correspond to the earlier
    # time limits of 0.1 s (bidding) and 1.0 s (replanning) on the development machine
    BID_EVALUATIONS = 2000
    REPLAN_EVALUATIONS = 20000

    def __init__(self, exp_policy, exp_env):
        super().__init__(exp_policy, exp_env)
        # the current epoffer which is under execution
        self.current_epoffer = None
        self.plan = []
        self.replan_needed = True # set to true if new ep accepted
    
    def on_ep_offer(self, message):
        """Bid the marginal cost of an offer that fits into the budget: the number of timesteps by which
        the EP lengthens our schedule. The currency of the market is one timestep of exploration budget."""
        content = message.content
        epoff = EPOffer(content["offer-id"], ep_from_content(content["ep"]), message.sender_name, content["prize"])
        super().on_ep_offer(message)
        self.epagent.expire_bids(self.rounds)
        canbid, bid = self.can_bid(epoff)
        if canbid:
            self.epagent.bid(epoff, self.rounds)
            self.outbox.append((message.sender_name, {"type": "ep-bid", "offer-id": epoff.offer_id, "value": bid}))

    def on_ep_award(self, message):
        """We won the offer: it becomes a commitment, which requires a replan"""
        self.epagent.won(message.content["offer-id"], message.content["price"])
        self.replan_needed = True

    def on_own_offer_sent(self, epoff):
        """We bid on our own offer like the others, without a message: we found the cluster, so we are often
        the nearest contractor"""
        canbid, bid = self.can_bid(epoff)
        if canbid:
            self.epagent.bid(epoff, self.rounds)
            self.epagent.receive_bid(epoff.offer_id, self.robot.name, bid)

    def on_self_award(self, epoff):
        """We won our own offer: it becomes a commitment, without an award message"""
        self.epagent.won(epoff.offer_id, epoff.bid_prize)
        self.replan_needed = True

    def report_completion(self, epoff):
        """We finished exploring a committed EP: report it to the offering agent, or, for our own offer,
        settle it in our own ledger without a message"""
        epoff = self.epagent.commitment_executed(epoff.offer_id, real_value = self.current_real_value)
        if epoff.offer_id in self.epagent.agreed_deals:
            self.epagent.offer_finished(epoff.offer_id, epoff.real_value)
        else:
            self.outbox.append((epoff.offering_agent_name, {"type": "ep-completed",
                "offer-id": epoff.offer_id, "real-value": epoff.real_value}))

    def inside_own_ep(self, x, y):
        """Whether a cell is inside the EP we are exploring or one of our commitments"""
        eps = [o.ep for o in self.epagent.commitments.values()]
        if self.current_epoffer is not None:
            eps.append(self.current_epoffer.ep)
        return any(ep.x_min <= x <= ep.x_max and ep.y_min <= y <= ep.y_max for ep in eps)

    def offer_own_streaks(self):
        """With the contractors offering, offer an EP around a streak of detections we found outside our EPs.
        Inside an EP, which we explore at the fine resolution anyway, the streak in progress is dropped."""
        if not self.contractors_offer or not self.observations:
            return
        obs = self.observations[-1]
        if self.inside_own_ep(obs["x"], obs["y"]):
            self.detections = []
            return
        streak = self.track_streak(obs)
        if streak is None:
            return
        offer = self.build_offer(streak)
        if offer is not None:
            self.make_offer(*offer)

    def offer_of(self, ep):
        """The committed offer of an exploration package of our plan"""
        return next(epoff for epoff in self.epagent.commitments.values() if epoff.ep is ep)

    def plan_ends_at(self):
        """Returns the location and time where the current plan ends - 
        this is the location where the next plan step should go"""
        if self.plan:
            planstep = self.plan[-1]
            if not isinstance(planstep, dict):
                print(f"Planstep {planstep}")
            xcurrent = planstep["x"]
            ycurrent = planstep["y"]
            t = planstep["t"] + 1 # one plus where it ends            
        else:
            xcurrent = self.robot.x
            ycurrent = self.robot.y
            t = self.timestep
        return xcurrent, ycurrent, t

    def replan(self):
        """Plan a path that covers the eps accepted but not terminated"""
        if not self.replan_needed:
            return        
        # Path:
        # [{"x":4, "y":4, "ep":None,}],....
        if not hasattr(self.robot, "oldplans"):
            self.robot.oldplans = {}
        
        self.robot.oldplans[self.timestep] = copy.copy(self.plan)
        oldplan = self.plan

        self.plan = []
        #
        # part one: copy the remainder of the current ep
        #
        step = None
        currentep = None
        if oldplan and oldplan[0]["ep"]:
            currentep = oldplan[0]["ep"]
            while True:
                step = oldplan.pop(0)
                if step["ep"] != currentep:
                    break
                self.plan.append(step)
        if step:
            print(f"Part one last step {step}")
        #
        # part two: create a plan accross the eps remaining
        #
        if any(x.ep is not currentep for x in self.epagent.commitments.values()):
            xcurrent, ycurrent, t = self.plan_ends_at()            
            print(f"Part two first step {xcurrent}, {ycurrent}, {t}")
            epset = ExplorationPackageSet()
            # epset.ep_to_explore += self.epagent.commitments
            # the remainder of the current ep is already in part one
            epset.ep_to_explore = [x.ep for x in self.epagent.commitments.values() if x.ep is not currentep]
            _, ep_path = epset.find_shortest_path_ep(start=[xcurrent, ycurrent], max_evaluations=self.REPLAN_EVALUATIONS)
            ep_xyplan = xyplan_from_ep_path(ep_path, t)
            self.plan += ep_xyplan
            print(f"Part two last step {self.plan[-1]}")
        #
        # part three: create random waypoints to the rest of the budget
        # 
        xcurrent, ycurrent, t = self.plan_ends_at()            
        print(f"Part three first step {xcurrent}, {ycurrent}, {t}")
        randxy = self.create_randxy(xcurrent=xcurrent, ycurrent=ycurrent, t=t)
        self.plan += randxy
        print(f"Part three last step {self.plan[-1]}")

        self.replan_needed = False
        self.current_epoffer = None
        self.current_real_value = 0 # accumulated real value for the offer
        # self.robot.oldplans[self.timestep] = copy.copy(self.plan)        

        if t > 1:
            print(f"Self plan first {self.plan[0]}, \n robot location {self.robot.x},{self.robot.y} \n last observation {self.observations[-1]}")
        return 
    
    def schedule_start(self):
        """Where and when a schedule through our other commitments starts: where the EP we are exploring
        ends, or our current position and time if we are not in an EP. Returns (x, y, t, current ep)."""
        if self.plan and self.plan[0]["ep"]:
            currentep = self.plan[0]["ep"]
            i = 0
            while i + 1 < len(self.plan) and self.plan[i + 1]["ep"] is currentep:
                i += 1
            return self.plan[i]["x"], self.plan[i]["y"], self.plan[i]["t"] + 1, currentep
        return self.robot.x, self.robot.y, self.timestep + 1, None

    def schedule_end(self, eps, start, t):
        """The time at which the shortest schedule from start, at time t, through the eps ends"""
        if not eps:
            return t
        epset = ExplorationPackageSet()
        epset.ep_to_explore = list(eps)
        _, ep_path = epset.find_shortest_path_ep(start, max_evaluations=self.BID_EVALUATIONS)
        return xyplan_from_ep_path(ep_path, t)[-1]["t"] + 1

    def can_bid(self, epoffer):
        """Whether we can bid for an offer, and our bid: (True, marginal cost) if the schedule through our
        commitments and the offered EP fits into the budget, (False, None) otherwise. The marginal cost is
        the number of timesteps by which the offered EP lengthens the schedule. The EPs we bid on and were
        not yet awarded count as commitments, so that winning several simultaneous auctions does not
        overcommit us."""
        x, y, t, currentep = self.schedule_start()
        held = [o.ep for o in self.epagent.commitments.values() if o.ep is not currentep]
        held += [o.ep for o in self.epagent.outstanding_bids.values()]
        t_without = self.schedule_end(held, [x, y], t)
        t_with = self.schedule_end(held + [epoffer.ep], [x, y], t)
        if t_with <= self.exp_policy["budget"]:
            return True, t_with - t_without
        return False, None

    def act(self, delta_t):
        """Call the following of the path"""
        self.timestep += delta_t
        self.epagent.expire_bids(self.rounds)
        self.replan()
        # just verify that the time is the right one
        if self.plan[0]["t"] != self.timestep:
            print(f"not asserted well! plan={self.plan[0]['t']} self.timestep={self.timestep}")
        assert self.plan[0]["t"] == self.timestep

        dist = abs(self.plan[0]['x']-self.robot.x) + abs(self.plan[0]['y']-self.robot.y)
        if dist > 2:
            print("Weird, big jump?")

        self.robot.add_action(f"loc [{self.plan[0]['x']}, {self.plan[0]['y']}]")

    

        # if the observation is a new one, add it to the value of the offer
        if self.current_epoffer:
            # FIXME: handle new observations, I will need to create a separate function for this, for the time being the value is zero, but should not be
            self.current_real_value += 0
        # terminating the current epoffer and marking it as executed
        # FIXME: self.plan[0]["ep"] supposed to fix when we are not in a new ep... but will this finish the last one?
        if self.current_epoffer and self.plan[0]["ep"] and self.current_epoffer.ep != self.plan[0]["ep"]:
            # current ep was terminated             
            self.report_completion(self.current_epoffer)
            self.current_epoffer = None
            self.current_real_value = 0
        # if the new one is the start of a new epoffer, start it
        if self.plan[0]["ep"] is not None: 
            self.current_epoffer = self.offer_of(self.plan[0]["ep"])
        # move on with the plan
        self.plan.pop(0)
        self.offer_own_streaks()
