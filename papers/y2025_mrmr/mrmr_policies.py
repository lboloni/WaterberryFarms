"""
mrmr_policies.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""

import numpy as np
import copy

from policy import Policy, AbstractCommunicateAndFollowPath
from communication import Message
from papers.y2025_mrmr.epmarket import EPM, EPAgent, EPOffer
from papers.y2025_mrmr.exploration_package import ExplorationPackage, ExplorationPackageSet
from papers.y2025_mrmr.xyplans import create_random_waypoints, xyplan_from_waypoints, xyplan_from_ep_path


class MRMR_Policy(Policy):
    """Implements the common ancestor of the MRMR (multiresolution multirobot models)"""

    def __init__(self, exp_policy, exp_env):
        self.exp_policy = exp_policy
        self.exp_env = exp_env
        self.name = exp_policy["policy-name"]
        self.timestep = -1 # local way to keep track of time, based on act calls
        # create the corresponding epagent and join the market
        self.epagent = EPAgent(self.name)
        self.epagent.policy = self
        self.epm = EPM().epm
        self.epm.join(self.epagent)
        self.observations = []
        # initializing the agent's random generator
        seed = self.exp_policy["seed"]
        self.random = np.random.default_rng(seed)

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

    def add_observation(self, obs):
        """MRMR policies collect the observations"""
        obs["name"] = self.name
        self.observations.append(obs)

class MRMR_Pioneer(MRMR_Policy):
    """Implements the Pioneer agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        super().__init__(exp_policy, exp_env)
        # contains the list of detections
        self.detections = []
        # when the streak ended, we put it here
        self.streak = None
        self.plan = []
        self.replan_needed = True
        self.offer_plan = None

    def replan(self):
        if not self.replan_needed:
            return    
        self.plan = []    
        randxy = self.create_randxy(xcurrent=self.robot.x, ycurrent=self.robot.y, t=self.timestep)
        self.plan += randxy
        self.replan_needed = False        

    def decide_to_offer_an_ep(self):
        """This is the function in which the agent decides to offer an ep. The ep is a border around the current streak of detections""" 
        if self.streak is None:
            return None
        streak = np.array(self.streak)
        # fixme
        im = self.robot.im
        geom_x_min = 0
        geom_x_max = im.width - 1 
        geom_y_min = 0
        geom_y_max = im.height - 1
        # create an ep around the streak
        x_min = np.min(streak[:,0])
        x_max = np.max(streak[:,0])
        y_min = np.min(streak[:,1])
        y_max = np.max(streak[:,1])
        # the longer the streak, the larger the ep
        # border = 3 * streak.shape[0]
        border = 3
        x_min = max(geom_x_min, x_min-border)
        x_max = min(geom_x_max, x_max+border)
        y_min = max(geom_y_min, y_min-border)
        y_max = min(geom_y_max, y_max+border)
        ep = ExplorationPackage(x_min, x_max, y_min, y_max, step = 2)
        # offer value is proportional with the area 
        offer_value = (x_max - x_min) * (y_max - y_min) * 0.1
        # reset the streak
        self.streak = None
        # check if the ep overlaps with any of the other eps.
        overlap = False
        for epold in self.epagent.agreed_deals:
            if epold.ep.overlap(ep):
                overlap = True
                break
        if overlap:
            return None
        else:
            return {"ep": ep, "value": offer_value}


    def act(self, delta_t):
        """The primary behavior of the agent. """
        self.timestep += delta_t
        self.replan()
        assert self.plan[0]["t"] == self.timestep
        self.robot.add_action(f"loc [{self.plan[0]['x']}, {self.plan[0]['y']}]")
        self.plan.pop(0) # move on with the plan
        # 
        # managing streaks of detections
        # FIXME: I am almost sure that this needs to be modified, tylcv 
        #
        if len(self.observations) == 0:
            return
        #
        # maintaining a streak???
        #
        obs = self.observations[-1]
        # print(f'Pio obs {obs["TYLCV"]["value"]}')
        if obs["TYLCV"]["value"] == 0.0:
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
        self.offer_plan = None
        for agentname in self.epm.agents:
            if agentname == self.name: continue
            agent = self.epm.agents[agentname]
            policy = agent.policy
            canbid, underbid = policy.can_bid(epoff)
            if canbid:
                # agent.bid(epoff, epoff.prize - underbid) # bid exactly the prize
                agent.bid(epoff, epoff.prize) # bid exactly the prize

        self.epm.clearing() # this will call agentC.won



class MRMR_Contractor(MRMR_Policy):
    """Implements the Contractor agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        super().__init__(exp_policy, exp_env)
        # the current epoffer which is under execution
        self.current_epoffer = None
        self.plan = []
        self.replan_needed = True # set to true if new ep accepted
    
    def won(self, epoff):
        """Called by the agent when the agent won the policy"""
        self.replan_needed = True

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
        if self.epagent.commitments:
            xcurrent, ycurrent, t = self.plan_ends_at()            
            print(f"Part two first step {xcurrent}, {ycurrent}, {t}")
            epset = ExplorationPackageSet()
            # epset.ep_to_explore += self.epagent.commitments
            epset.ep_to_explore = [x.ep for x in self.epagent.commitments]
            _, ep_path = epset.find_shortest_path_ep(start=[xcurrent, ycurrent], maxtime=1.0)
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
    
    def can_bid(self, epoffer):
        """This function allows the agent to decide whether it can bid for a certain offer or not."""
        # step one: find the termination time of the current ep
        t = self.timestep
        currentep = None
        if self.plan[0]["ep"]: # we are in an ep
            i = 0
            currentep = self.plan[0]["ep"]
            while True:
                t = self.plan[i]["t"]
                i+=1
                if self.plan[i]["ep"] != currentep:
                    break
        t = t + 1
        # get all the eps, except the current one
        # eps = copy.copy(self.epagent.commitments)
        eps = ExplorationPackageSet()
        eps.ep_to_explore = [x.ep for x in self.epagent.commitments if x.ep != currentep]
        eps.add_ep(epoffer.ep)
        # create an optimal path 
        current = [self.robot.x, self.robot.y]
        _, best_ep_path = eps.find_shortest_path_ep(current, maxtime=1.0)
        path = xyplan_from_ep_path(best_ep_path, t)
        if path[-1]["t"] < self.exp_policy["budget"]:
            # FIXME: this needs to be made more sophisticated
            # underbid with one tenth of the budget remaining
            underbid = (self.exp_policy["budget"] - path[-1]["t"]) / 10.0
            return True, underbid # we can bid
        return False, 0 # we cannot bid

    def act(self, delta_t):
        """Call the following of the path"""
        self.timestep += delta_t
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
            self.epagent.commitment_executed(self.current_epoffer, real_value = self.current_real_value)
            self.current_epoffer = None
            self.current_real_value = 0
        # if the new one is the start of a new epoffer, start it
        if self.plan[0]["ep"] is not None: 
            self.current_epoffer = self.epm.ep_to_offer[self.plan[0]["ep"]]
        # move on with the plan
        self.plan.pop(0)
