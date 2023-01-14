import numpy as np
import ast
import math
import random
import itertools
import logging
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from Robot import Robot

class Policy:
    def __init__(self):
        """Create a policy that is not yet assigned to the robot"""
        self.env = None

    def assign(self, robot):
        """Assign the policy to a specific robot"""
        self.robot = robot
        
    def act(self, delta_t):
        """Act over the time of delta T. When implemented, it should add an action to be executed to
        the robot's action queue - it is not supposed to execute it here"""        
        pass
    
    def __str__(self):
        return self.__class__.__name__
    
    
class AbstractWaypointPolicy(Policy):
    """The ancestor of all the policies that are based on choosing waypoints"""
        
    def move_towards_location(self, targetx, targety, vel, delta_t):
        """Schedule the actions to move the robot towards a target location. Returns true 
        if the robot will reach that location as the result of this step and false otherwise"""
        deltax = targetx - self.robot.x
        deltay = targety - self.robot.y
        veclength = np.linalg.norm([deltax, deltay])
        logging.debug(f"veclength {veclength}")
        if veclength == 0: # if we are there, stay there 
            self.robot.add_action("vel [0, 0]")
            return True
        if veclength <= self.vel * delta_t: # if we are closer than the velocity, then just go there
            logging.debug(f"We are there so go there [{targetx}, {targety}]")
            self.robot.add_action("vel [0, 0]")
            self.robot.add_action(f"loc [{targetx}, {targety}]")
            return True
        # move towards the location with the specified velocity
        #vel_x = min(vel * deltax / veclength, delta_t * deltax)
        #vel_y = min(vel * deltay / veclength, delta_t * deltay)
        vel_x = vel * deltax / veclength
        vel_y = vel * deltay / veclength
        action = f"vel [{vel_x}, {vel_y}]"
        logging.debug(f"move_towards_location setting action = {action}")
        self.robot.add_action(action)
        return False
    
    
class GoToLocationPolicy(AbstractWaypointPolicy):
    """This is a simple policy that make the robot go to a certain location."""
    def __init__(self, locx, locy, vel):
        super().__init__()
        self.locx, self.locy, self.vel = locx, locy, vel
        
    def act(self, delta_t):
        """ Head towards locx, locy with a velocity vel"""
        self.move_towards_location(self.locx, self.locy, self.vel, delta_t)
        
        
class FollowPathPolicy(AbstractWaypointPolicy):
    """A policy that makes a robot follow a certain path specified as a series 
    of waypoints. If repeat is true, it repeats the path indefinitely.
    """
    def __init__(self, vel, waypoints, repeat = False):
        super().__init__()
        self.waypoints = waypoints.copy()
        self.vel = vel
        self.currentwaypoint = 0
        self.repeat = repeat
        
    def act(self, delta_t):
        ## if we are done with the movement, don't do anything
        if self.currentwaypoint == -1:
            return
        ## move towards the current waypoint
        wp = self.waypoints[ self.currentwaypoint ]
        done = self.move_towards_location(wp[0], wp[1], self.vel, delta_t)
        if done:
            self.currentwaypoint = self.currentwaypoint + 1
            if self.currentwaypoint == len(self.waypoints):
                if self.repeat: 
                    self.currentwaypoint = 0
                else:
                    self.currentwaypoint = -1
                    
                    
class RandomWaypointPolicy(AbstractWaypointPolicy):
    """A policy that makes the robot follow a random waypoint behavior within a 
    specified region using a constant velocity. The region is specied with the 
    low_point and high_point each of them having (x,y) formats """
    def __init__(self, vel, low_point, high_point, seed):
        super().__init__()
        self.random = np.random.default_rng(seed)
        self.vel = vel
        self.low_point = low_point
        self.high_point = high_point
        self.nextwaypoint = None
                
    def act(self, delta_t):
        """Moves towards the chosen waypoint. If the waypoint is reached, it
        picks the next waypoint"""
        if self.nextwaypoint == None:
            x = self.random.uniform(self.low_point[0], self.high_point[0])
            y = self.random.uniform(self.low_point[1], self.high_point[1])
            self.nextwaypoint = [x, y]
        done = self.move_towards_location(self.nextwaypoint[0], self.nextwaypoint[1], self.vel, delta_t)
        if done:
            self.nextwaypoint = None
            
            
class InformationGreedyPolicy(AbstractWaypointPolicy):
    """A policy which makes the robot choose its next waypoint to be the one 
    with the largest information value from an square area of radius span 
    around the current location"""
    def __init__(self, vel, span = 5):
        super().__init__()
        self.vel = vel
        self.nextwaypoint = None
        self.span = span
                
    def act(self, delta_t):
        """Moves towards the chosen waypoint. If the waypoint had been reached, chooses the next waypoint
        which is the one with the highest uncertainty value"""
        if self.nextwaypoint == None:
            feasible_waypoints = self.generate_feasible_waypoints()
            waypoint_values = [self.robot.im.uncertainty[x[0],x[1]] for x in feasible_waypoints]       
            bestindex = np.argmax(waypoint_values)            
            self.nextwaypoint = feasible_waypoints[bestindex]
        done = self.move_towards_location(self.nextwaypoint[0], self.nextwaypoint[1], self.vel, delta_t)
        if done:
            self.nextwaypoint = None
            
    def generate_feasible_waypoints(self):
        # generate all the feasible waypoints: the points in a rectangular area 
        # of extent span from the current points
        currentx = int(self.robot.x)
        currenty = int(self.robot.y)
        rangex = range(max(0, currentx - self.span), min(currentx+self.span, self.robot.env.width))
        rangey = range(max(0, currenty - self.span), min(currenty+self.span, self.robot.env.height))
        val = itertools.product(rangex, rangey)
        return list(val)            
            
