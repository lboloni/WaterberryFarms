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
    def __init__(self, name, env, robot):
        self.name = name
        self.env = env
        self.robot = robot
        
    def act(self, delta_t):
        """Act over the time of delta T. When implemented, it should add an action to be executed to
        the robot's action queue - it is not supposed to execute it here"""        
        pass
    
    def __str__(self):
        return self.name
    
    
class AbstractWaypointPolicy(Policy):
    """The ancestor of all the policies that are based on choosing waypoints"""
    
    ##def __init__(self, name, env, robot):
    ##    super().__init__("AbstractWaypointPolicy", env, robot)
    
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
    def __init__(self, env, robot, locx, locy, vel):
        super().__init__("GoToPolicy", env, robot)
        self.locx, self.locy, self.vel = locx, locy, vel
        
    def act(self, delta_t):
        """ Head towards locx, locy with a velocity vel"""
        ##logging.debug("HERE")
        self.move_towards_location(self.locx, self.locy, self.vel, delta_t)
        
        
class FollowPathPolicy(AbstractWaypointPolicy):
    """A policy that makes a robot follow a certain path specified as a series of waypoints. 
    If repeat is true, it repeats the path indefinitely."""
    def __init__(self, env, robot, vel, waypoints, repeat = False):
        super().__init__("FollowPathPolicy", env, robot)
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
    """A policy that makes the robot follow a random waypoint behavior within a specified region
    using a constant velocity.
    The region is specied with the low_point and high_point each of them having (x,y) formats """
    def __init__(self, env, robot, vel, low_point, high_point, seed):
        super().__init__("RandomWaypointPolicy", env, robot)
        self.random = np.random.default_rng(seed)
        self.vel = vel
        self.low_point = low_point
        self.high_point = high_point
        self.nextwaypoint = None
                
    def act(self, delta_t):
        """Moves towards the chosen waypoint. If the waypoint is """
        if self.nextwaypoint == None:
            x = self.random.uniform(self.low_point[0], self.high_point[0])
            y = self.random.uniform(self.low_point[1], self.high_point[1])
            self.nextwaypoint = [x, y]
        done = self.move_towards_location(self.nextwaypoint[0], self.nextwaypoint[1], self.vel, delta_t)
        if done:
            self.nextwaypoint = None
            
            
class InformationGreedyPolicy(AbstractWaypointPolicy):
    """A policy which makes the robot choose its next waypoint to be the one with the 
    largest information value from an square area of radius span around the current location"""
    def __init__(self, robot, vel, span = 5):
        super().__init__("InformationGreedyPolicy", robot.env, robot)
        self.vel = vel
        self.nextwaypoint = None
        self.span = span
                
    def act(self, delta_t):
        """Moves towards the chosen waypoint. If the waypoint had been reached, chooses the next waypoint
        which is the one with the highest uncertainty value"""
        if self.nextwaypoint == None:
            feasible_waypoints = self.generate_feasible_waypoints()
            # waypoint_values = [self.robot.im.estimate_voi(x) for x in feasible_waypoints]            
            # tcurrent = time.perf_counter()
            waypoint_values = [self.robot.im.uncertainty[x[0],x[1]] for x in feasible_waypoints]       
            # print(f"time spent here {time.perf_counter() - tcurrent}")
            bestindex = np.argmax(waypoint_values)            
            self.nextwaypoint = feasible_waypoints[bestindex]
        done = self.move_towards_location(self.nextwaypoint[0], self.nextwaypoint[1], self.vel, delta_t)
        if done:
            self.nextwaypoint = None
            
    def generate_feasible_waypoints(self):
        # generate all the feasible waypoints: the points in a rectangular area of extent span from 
        # the current points
        currentx = int(self.robot.x)
        currenty = int(self.robot.y)
        rangex = range(max(0, currentx - self.span), min(currentx+self.span, self.robot.env.width))
        rangey = range(max(0, currenty - self.span), min(currenty+self.span, self.robot.env.height))
        val = itertools.product(rangex, rangey)
        return list(val)            
            
def generate_lawnmower(xmin, xmax, ymin, ymax, winds):
    """Generates a horizontal lawnmower path on the list """
    current = [xmin, ymin]
    ystep = (ymax - ymin) / (2 * winds)
    path = []
    path.append(current)
    for i in range(winds):
        path.append([xmax, current[1]])
        path.append([xmax, current[1]+ystep])
        path.append([xmin, current[1]+ystep])
        current = [xmin, current[1]+2 * ystep]
        path.append(current)
    path.append([xmax, current[1]])
    return path


###########################################
def updated_generate_lawnmower(xmin, xmax, xStep, ymin, ymax, yStep, epidemicPoints, coverageDistance):
    """Generates a horizontal lawnmower path on the list """
    epidemicPoints= sorted(epidemicPoints, key=lambda x: (x[1], x[0]), reverse= True)
    noEpidemicPoints= len(epidemicPoints)
    epidemicPointsToBeAdded=[]
    path = []
    direction=1
    largestX=0.0
    for y in np.arange(ymin, ymax+yStep, yStep):
        if (direction==1):
            for x in np.arange(xmin, xmax + xStep, xStep):
                path.append([x, y])
                while (y <= epidemicPoints[noEpidemicPoints-1][1] and x <= epidemicPoints[noEpidemicPoints-1][0] and \
                        y + yStep >= epidemicPoints[noEpidemicPoints-1][1] and x + xStep >= epidemicPoints[noEpidemicPoints-1][0]):
                    if (noEpidemicPoints!=0 and EpidemicPointsNeedToBeInserted((x,y), epidemicPoints[noEpidemicPoints-1], coverageDistance, xStep, yStep)):
                        epidemicPointsToBeAdded.append(epidemicPoints[noEpidemicPoints-1])
                        path.append(epidemicPoints[noEpidemicPoints - 1])
                    noEpidemicPoints=noEpidemicPoints-1


                largestX=x
            direction=-1
        else:
            for x in np.arange(largestX, xmin-xStep, -xStep):
                path.append([x, y])
                while (y <= epidemicPoints[noEpidemicPoints - 1][1] and x <= epidemicPoints[noEpidemicPoints - 1][0] and \
                        y + yStep >= epidemicPoints[noEpidemicPoints - 1][1] and x + xStep >=
                        epidemicPoints[noEpidemicPoints - 1][0]):

                    if (noEpidemicPoints!=0 and EpidemicPointsNeedToBeInserted((x,y), epidemicPoints[noEpidemicPoints-1], coverageDistance, xStep, yStep)):
                        epidemicPointsToBeAdded.append(epidemicPoints[noEpidemicPoints-1])
                        path.append(epidemicPoints[noEpidemicPoints - 1])
                    noEpidemicPoints=noEpidemicPoints-1

            direction=1

    return path

def EpidemicPointsNeedToBeInserted(pathPoint, epidemicPoint, coverageDistance, xStep, yStep):
    if (CalculateEuclideanDistance(pathPoint, epidemicPoint) >= coverageDistance):
        if (CalculateEuclideanDistance((pathPoint[0] + xStep, pathPoint[1]), epidemicPoint) >= coverageDistance):
            if (CalculateEuclideanDistance((pathPoint[0], pathPoint[1] + yStep),
                                           epidemicPoint) >= coverageDistance):
                if (CalculateEuclideanDistance((pathPoint[0] + xStep, pathPoint[1] + yStep),
                                               epidemicPoint) >= coverageDistance):
                    return True
    return False


def CalculateEuclideanDistance(point1, point2):
    return math.sqrt(math.pow(point2[0]-point1[0],2) + math.pow(point2[1]-point1[1],2))

def visualize_path_and_epidemic_points(pointlist, epidemicPoints):
    """Visualize the values for Sam Matloob-s code"""
    # plot the generated path
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    # add the trajectory
    path = Path(pointlist)
    ax.add_patch(patches.PathPatch(path, fill=False))
    # add the epidemic points
    for ep in epidemicPoints:
        ax.add_patch(patches.Circle(ep, radius=0.3))
    ax.set_xlim(-2, 20)
    ax.set_ylim(-2, 20)
    plt.show()
    ax.plot(test)

if __name__ == "__main__":
    # Testing Sam Matloob's code 
    epidemicPoints = [(8, 5), (9, 3), (5, 2), (3,0),(6,0)]
    # path= updated_generate_lawnmower(0, 9, 5, 0, 9, 5, epidemicPoints, 1)
    path= updated_generate_lawnmower(0, 9, 2, 0, 9, 2, epidemicPoints, 1)
    visualize_path_and_epidemic_points(path, epidemicPoints)
    #print(test)



test= updated_generate_lawnmower(0, 9, 5, 0, 9, 5, [(8, 5), (9, 3), (5, 2), (3,0),(6,0)], 1)

# path generated: [[0, 0], (3, 0), [5, 0], (6, 0), (5, 2), (9, 3), (8, 5), [10, 0], [10, 5], [5, 5], [0, 5], [0, 10], [5, 10], [10, 10]]