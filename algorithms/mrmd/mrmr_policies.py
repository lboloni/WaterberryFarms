"""
mrmr_policies.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""
from policy import AbstractCommunicateAndFollowPath
from communication import Message
from algorithms.mrmd.epmarket import EPM, EPAgent, EPOffer
from algorithms.mrmd.exploration_package import ExplorationPackage
import numpy as np

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


def create_random_waypoints(exp, xcurrent, ycurrent, budget):
    """Create a random waypoint path, whose area is described in 
    exp['area'] and minimum lenghth in 'budget'"""
    # x = self.random.uniform(self.low_point[0], self.high_point[0])
    # y = self.random.uniform(self.low_point[1], self.high_point[1])
    # self.nextwaypoint = [x, y]


class MRMR_Pioneer(AbstractCommunicateAndFollowPath):
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
        if obs["value"] = 1:
            self.detections.append([obs["x"], obs["y"]])
        else: # end of streak
            if len(self.detections) > 0:
                self.streak = self.detections
                self.detections = []
                self.ep = self.decide_to_offer_an_ep()        

    def act_send(self, round):
        """FIXME: the market negotiation is supposed to happen here, with
        messages, but for the time being we assume that we have direct access to the market"""
        # print(f"{self.name} act_send called at round {round}")
        if round == 0:
            if self.ep is not None:
                # sending a message
                # self.robot.com.send(self.robot, destination=None, message = Message(ep))
                # directly accessing the market
                # assume a fixe
                epoff = EPOffer(self.ep, self.name, 100)
                self.epm.add_offer(epoff)    
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")

class MRMR_Contractor(AbstractCommunicateAndFollowPath):
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
    
    def act(self, delta_t):
        """Call the following of the path"""
        super().act(delta_t)

    def act_send(self, round):
        print(f"{self.name} act_send called at round {round}")
        self.robot.com.send(self.robot, destination=None, message = Message("hello"))
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")
