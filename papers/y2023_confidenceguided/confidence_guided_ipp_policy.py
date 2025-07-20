"""
confidence_guided_ipp_policy.py

"""
from policy import AbstractWaypointPolicy
import itertools
import numpy as np


class AbstractEmbeddedEstimatorPolicy(AbstractWaypointPolicy):
    """A policy that is using an embedded estimator"""
    def __init__(self, internal_estimator = None):
        self.internal_estimator = internal_estimator

    def get_im(self):
        return self.internal_estimator

    def add_observation(self, obs):
        """Pass the observation to the internal estimator"""
        ### test is it depends on the observation
        #obs2 = copy.deepcopy(obs)
        #print(obs2)
        #obs2["TYLCV"]["value"] = 0
        #self.internal_estimator.add_observation(obs2)
        self.internal_estimator.add_observation(obs)

    def im_proceed(self, delta_t):
        self.internal_estimator.proceed(delta_t)

class ConfidenceGuidedPathPlanning(AbstractEmbeddedEstimatorPolicy):
    """A policy which makes the robot choose its next waypoint to be the one 
    with the largest information value from an square area of radius span 
    around the current location"""
    def __init__(self, vel, low_point, high_point, estimator, span = 5, seed = 0):
        super().__init__(estimator)
        self.vel = vel
        self.low_point = low_point
        self.high_point = high_point
        self.next_waypoint = None
        self.span = span
        self.random = np.random.default_rng(seed)
        self.delta_im = 0

    def act(self, delta_t):
        """Moves towards the chosen waypoint. If the waypoint had been reached, chooses the next waypoint
        which is the one with the highest uncertainty value"""
        self.delta_im = self.delta_im + delta_t        
        if self.next_waypoint is None:
            self.im_proceed(self.delta_im)
            self.delta_im = 0
            feasible_waypoints = self.generate_feasible_waypoints()
            self.next_waypoint = self.choose_next_waypoint(feasible_waypoints)
        done = self.move_towards_location(self.next_waypoint[0], self.next_waypoint[1], self.vel, delta_t)
        if done:
            self.next_waypoint = None

    def choose_next_waypoint_exp(self, feasible_waypoints):
        """choose the next waypoint to head to. Currently it uses the im_tylcv to go, but this could be made more generic"""
        ##waypoint_values = [self.get_im().im_tylcv.uncertainty[x[0],x[1]] + 0.2 * self.get_im().im_tylcv.value[x[0], x[1]] for x in feasible_waypoints]      
        # a little trick, bringing in the mask
        waypoint_values = []
        for x in feasible_waypoints:
            uncer = self.get_im().im_tylcv.uncertainty[x[0],x[1]]
            val = self.get_im().im_tylcv.value[x[0],x[1]]
            if x[0] > 15:
                value = uncer * 5.0
            else: 
                value = uncer * 1.0
            # * (1 - self.get_im().im_tylcv.value[x[0], x[1]])
            waypoint_values.append(value)
        #  bestindex = np.argmax(waypoint_values)
        bestvalue = np.max(waypoint_values)
        accumulate = []
        for x, value in zip(feasible_waypoints, waypoint_values):
            if value == bestvalue:
                accumulate.append(x)
        # self.next_waypoint = feasible_waypoints[bestindex]
        next_waypoint = self.random.choice(accumulate)
        print(f"CGP: next waypoint {next_waypoint}")
        return next_waypoint



    def choose_next_waypoint(self, feasible_waypoints):
        """choose the next waypoint to head to. Currently it uses the im_tylcv to go, but this could be made more generic"""
        waypoint_values = [self.get_im().im_tylcv.uncertainty[x[0],x[1]] for x in feasible_waypoints]       
        # bestindex = np.argmax(waypoint_values)
        bestvalue = np.max(waypoint_values)
        accumulate = []
        for x, value in zip(feasible_waypoints, waypoint_values):
            if value == bestvalue:
                accumulate.append(x)
        # self.next_waypoint = feasible_waypoints[bestindex]
        next_waypoint = self.random.choice(accumulate)
        print(f"CGP: next waypoint {next_waypoint}")
        return next_waypoint

    def generate_feasible_waypoints(self):
        """generate all the feasible waypoints: the points in a rectangular 
        area of extent span from the current points, from which we remove the current waypoint.
        """
        currentx = int(self.robot.x)
        currenty = int(self.robot.y)
        rangex = range(max(self.low_point[0], currentx - self.span), min(currentx+self.span, self.high_point[0]))
        rangey = range(max(self.low_point[1], currenty - self.span), min(currenty+self.span, self.high_point[1]))
        val = itertools.product(rangex, rangey)
        # remove the current location from it
        retval = list(val)            
        retval.remove((self.robot.x, self.robot.y))
        return retval            
