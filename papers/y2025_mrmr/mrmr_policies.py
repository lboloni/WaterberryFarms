"""
mrmr_policies.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""
from policy import Policy
from communication import Message
from algorithms.mrmd.epmarket import EPM, EPAgent, EPOffer
from algorithms.mrmd.exploration_package import ExplorationPackage
import numpy as np

class MRMR_Pioneer(Policy):
    """Implements the Pioneer agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        # how to get the budget?
        #waypoints = create_random_waypoints(exp, self.robot)
        #repeat = False
        vel = 1
        #super().__init__(vel, waypoints, repeat)
        super().__init__(vel, None, repeat=False)
        self.exp_policy = exp_policy
        self.exp_env = exp_env
        self.name = exp_policy["policy-name"]
        self.epagent = EPAgent(self.name)
        self.epm = EPM().epm
        print("MRMR_Pioneer policy was created")
        # contains the list of detections
        self.detections = []
        # when the streak ended, we put it here
        self.streak = None
    
    def decide_to_offer_an_ep(self):
        """This is the function in which the agent decides to offer an ep""" 
        if self.streak is None:
            return None
        # fixme
        geom = exp_env.geometry()
        # create an ep around the streak
        x_min = np.min(self.streak[:,0])
        x_max = np.max(self.streak[:,0])
        y_min = np.min(self.streak[:,1])
        y_max = np.max(self.streak[:,1])
        # the longer the streak, the larger the ep
        border = 3 * self.streak.shape[0]
        x_min = max(geom.x_min, x_min-border)
        x_max = min(geom.x_max, x_max-border)
        y_min = max(geom.x_min, y_min-border)
        y_max = min(geom.x_min, y_max-border)
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
            return ep

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
                self.ep = self.decide_to_offer_an_ep()
        if self.ep is None:
            return
        #
        # participation in the market, if we decided to make an offer
        #
        


class MRMR_Contractor(Policy):
    """Implements the Contractor agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        super().__init__(vel, waypoints, repeat)
        self.name = exp_policy["policy-name"]
        self.epagent = EPAgent(self.name)
        self.epm = EPM().epm
        print("MRMR_Contractor policy was created")
        # the current epoffer which is under execution
        self.current_epoffer = None
        self.replan_needed = True # set to true if new ep accepted
        self.plan = self.replan()
    
    def create_ep_xyplan(self, eps, time):
        """Create the xyplan for the eps. Mark each segment with the specific ep, and the intermediate ones with no ep."""
        # FIXME: implement me
        return []

    def create_random_waypoint_xyplan(self, time):
        """Create a random waypoint xy plan starting from time to the time 
        described in the budget of the policy"""
        # FIXME: implement me
        return []

    def replan(self):
        """Plan a path that covers the eps accepted but not terminated"""
        if not self.replan_needed:
            return
        # FIXME implement me
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
        t = self.plan[-1]["t"]
        eps = self.epagent.commitments
        ep_plan = self.create_ep_plan(eps, t+1)
        self.plan.append(ep_plan)
        # part three: create random waypoints to the rest
        # FIXME implement me
        self.replan_needed = False
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
        # super().act(delta_t)
        # just verify that the time is the right one
        assert self.plan[0]["t"] != self.robot.env.t
        self.robot.add_action(f"loc [{self.plan[0]['x']}, {self.plan[0]['y']}]")
        # set the current location


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