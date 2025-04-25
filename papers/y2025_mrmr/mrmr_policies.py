"""
mrmr_policies.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""
from policy import Policy, AbstractCommunicateAndFollowPath
from communication import Message
from papers.y2025_mrmr.epmarket import EPM, EPAgent, EPOffer
from papers.y2025_mrmr.exploration_package import ExplorationPackage, ExplorationPackageSet
from papers.y2025_mrmr.xyplans import create_random_waypoints, xyplan_from_waypoints, xyplan_from_ep_path
import numpy as np

class MRMR_Pioneer(Policy):
    """Implements the Pioneer agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        super().__init__()
        self.exp_policy = exp_policy
        self.exp_env = exp_env
        self.name = exp_policy["policy-name"]
        # create the corresponding epagent and join the market
        self.epagent = EPAgent(self.name)
        self.epagent.policy = self
        self.epm = EPM().epm
        self.epm.join(self.epagent)
        # contains the list of detections
        self.detections = []
        # when the streak ended, we put it here
        self.streak = None
    
    def decide_to_offer_an_ep(self):
        """This is the function in which the agent decides to offer an ep""" 
        if self.streak is None:
            return None
        # fixme
        env = self.robot.env
        geom_x_min = 0
        geom_x_max = env.width
        geom_y_min = 0
        geom_y_max = env.height
        # create an ep around the streak
        x_min = np.min(self.streak[:,0])
        x_max = np.max(self.streak[:,0])
        y_min = np.min(self.streak[:,1])
        y_max = np.max(self.streak[:,1])
        # the longer the streak, the larger the ep
        border = 3 * self.streak.shape[0]
        x_min = max(geom_x_min, x_min-border)
        x_max = min(geom_x_max, x_max-border)
        y_min = max(geom_y_min, y_min-border)
        y_max = min(geom_y_max, y_max-border)
        ep = ExplorationPackage(x_min, x_max, y_min, y_max, step = 2)
        # reset the streak
        self.streak = None
        # check if the ep overlaps with any of the other eps.
        overlap = False
        for epold in self.epagent.agreed_deals:
            if epold.overlap(ep):
                overlap = True
                break
        if overlap:
            return None
        else:
            return {"ep": ep, "value": 100}

    def act(self, delta_t):
        """The primary behavior of the agent. """
        if self.waypoints is None:
            budget = 1000 # fixme, we need to take this from somewhere
            self.waypoints = create_random_waypoints(self.exp_policy, self.robot.x, self.robot.y, budget)
        super().act(delta_t)
        # if this is an observation with a detection, add it to detections
        obs = self.observations[-1]
        if obs["value"] == 1:
            self.detections.append([obs["x"], obs["y"]])
        else: # end of streak
            if len(self.detections) > 0:
                self.streak = self.detections
                self.detections = []
                self.offer_plan = self.decide_to_offer_an_ep()
        if self.offer_plan is None:
            return
        #
        # participation in the market, if we decided to make an offer
        #
        epoff = self.epagent.offer(self.offer_plan["ep"], self.offer_plan["value"])
        # FIXME: continue the market participation play here from MRMR-experiments.ipynb
        for agentname in self.epm.agents:
            if agentname == self.name: continue
            agent = self.epm.agents[agentname]
            policy = agent.policy
            if policy.can_bid(epoff):
                agent.bid(epoff, 100) # for the time being, bid exactly
        self.epm.clearing() # this will call agentC.won



class MRMR_Contractor(Policy):
    """Implements the Contractor agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        super().__init__()
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        self.exp_policy = exp_policy
        self.exp_env = exp_env
        self.name = exp_policy["policy-name"]
        # create the corresponding epagent and join the market
        self.epagent = EPAgent(self.name)
        self.epagent.policy = self
        self.epm = EPM().epm
        self.epm.join(self.epagent)
        # the current epoffer which is under execution
        self.current_epoffer = None
        self.replan_needed = True # set to true if new ep accepted
        self.plan = []
        self.replan()
    
    def won(self, epoff):
        """Called by the agent when the agent won the policy"""
        self.replan_needed = True


    def create_ep_plan(eps, t):
        """Creates a plan that contains segments"""

    def replan(self):
        """Plan a path that covers the eps accepted but not terminated"""
        if not self.replan_needed:
            return
        # Path:
        # [{"x":4, "y":4, "ep":None,}],....
        oldplan = self.plan
        self.plan = []
        # part one: copy the remainder of the current ep
        if len(oldplan) > 0 and oldplan[0]["ep"] is not None:
            currentep = oldplan[0]["ep"]
            while True:
                step = oldplan.pop()
                if step["ep"] != currentep:
                    break
                self.plan.append(step)
        # part two: create a plan accross the eps remaining

        if len(self.plan) == 0:
            xcurrent = self.robot.x
            ycurrent = self.robot.y
        else:
            xcurrent = self.plan[-1]["x"]
            ycurrent = self.plan[-1]["y"]

        t = self.plan[-1]["t"]
        epset = ExplorationPackageSet()
        epset.ep_to_explore += self.epagent.commitments
        _, ep_path = epset.find_shortest_path_ep(start=[xcurrent, ycurrent])
        ep_path = self.create_ep_plan(epset, t+1)
        ep_xyplan = xyplan_from_ep_path(ep_path, t+1)
        self.plan.append(ep_xyplan)
        # part three: create random waypoints to the rest
        t = self.plan[-1]["t"]
        # fixme, this is some code to chain the xyplans
        if len(self.plan) == 0:
            xcurrent = self.robot.x
            ycurrent = self.robot.y
        else:
            xcurrent = self.plan[-1]["x"]
            ycurrent = self.plan[-1]["y"]
        xmin = 0 # fixme, extract from the environment
        xmax = 100
        ymin = 0
        ymax = 100
        budget = self.exp_policy["budget"] - t
        seed = self.exp_policy["seed"]
        randwp = create_random_waypoints(seed, xcurrent, ycurrent, xmin, xmax, ymin, ymax, budget)
        randxy = xyplan_from_waypoints(randwp, t, vel=1, ep=None)
        self.plan += randxy
        self.replan_needed = False
        self.current_epoffer = None
        self.current_real_value = 0 # accumulated real value for the offer
        return 
    
    def can_bid(self, epoffer):
        """This function allows the agent to decide whether it can bid for a certain offer or not."""
        # step one: find the termination time of the current ep
        t = self.robot.env.t
        if self.plan[0]["ep"] is not None: # we are in an ep
            i = 0
            currentep = self.plan[0]["ep"]
            while True:
                t = self.plan[i]["t"]
                i+=1
                if self.plan[i]["ep"] != currentep:
                    break
        t = t + 1
        # get all the eps, except the current one
        eps = self.epagent.commitments
        eps.remove(currentep)
        eps.add(epoffer.ep)
        # create an optimal path 
        path = self.create_ep_xyplan(eps, t)
        if path[-1]["t"] < self.exp["budget"]:
            return True # we can bid
        return False # we cannot bid

    def act(self, delta_t):
        """Call the following of the path"""
        # just verify that the time is the right one
        assert self.plan[0]["t"] != self.robot.env.t
        self.robot.add_action(f"loc [{self.plan[0]['x']}, {self.plan[0]['y']}]")
        # if the observation is a new one, add it to the value of the offer
        if self.current_epoffer is not None:
            # FIXME: handle new observations, I will need to create a separate function for this, for the time being the value is zero, but should not be
            self.current_real_value += 0
        # terminating the current epoffer
        if self.current_epoffer != self.plan[0]["ep"]:
            # current ep was terminated             
            self.epagent.commitment_executed(self.current_epoffer, real_value = self.current_real_value)
            self.current_epoffer = None
            self.current_real_value = 0
        # if the new one is the start of a new epoffer, start it
        if self.plan[0]["ep"] is not None:
            self.current_epoffer = self.plan[0]["ep"]
        # move on with the plan
        self.plan.pop(0)

class SimpleCommunicator(AbstractCommunicateAndFollowPath):
    """Example, simple communicator policy"""
    def __init__(self, exp_policy, exp_env):
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        super().__init__(vel, waypoints, repeat)
        self.name = exp_policy["policy-name"]
    
    def act(self, delta_t):
        """Call the following of the path"""
        super().act(delta_t)

    def act_send(self, round):
        print(f"{self.name} act_send called at round {round}")
        self.robot.com.send(self.robot, destination=None, message = Message("hello"))
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")