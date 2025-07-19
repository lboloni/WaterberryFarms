"""
xyplans.py

Functions that implement fully unrolled plans for the policy of a robot. 
"""

from path_generators import get_path_length
from policy import FollowPathPolicy
from robot import Robot
import numpy as np


def xyplan_from_waypoints(waypoints, t=0, vel=1, ep=None):
    """creates an xy plan from a set of waypoints. We can do it with a fictional robot"""
    plan = []
    robot = Robot("r", waypoints[0][0], waypoints[0][1], 0)
    policy = FollowPathPolicy(vel,waypoints)
    policy.assign(robot)
    robot.assign_policy(policy)
    while policy.currentwaypoint != -1:
        robot.enact_policy(1.0)
        robot.proceed(1.0)
        # print(robot)
        step = {}
        step["t"] = t
        t += 1
        step["x"] = robot.x
        step["y"] = robot.y
        step["ep"] = ep
        plan.append(step)
    return plan

def create_random_waypoints(random, xcurrent=0, ycurrent=0, xmin=0, xmax=100, ymin=0, ymax=100, budget=1000, vel=1):
    """Create a random waypoint path, whose area is described in 
    exp['area'] and minimum length in 'budget' when traversed with velocity vel"""
    waypoints = [[xcurrent, ycurrent]]
    while True:
        x = int(random.uniform(xmin, xmax))
        y = int(random.uniform(ymin, ymax))
        waypoints.append([x,y])
        if get_path_length(waypoints) > budget/vel:
            break
    return waypoints

def xyplan_from_ep_path(ep_path, t):
    """Create the xyplan from the ep path, which is a list of components that might or might not have an ep
    """
    full_plan = []
    for i in range(len(ep_path)):
        epsegment = ep_path[i]
        waypoints = epsegment["path"]
        ep = epsegment["ep"]
        plan = xyplan_from_waypoints(waypoints, t, vel=1, ep=ep)
        full_plan += plan
        t += len(plan)
        if i < len(ep_path)-1: # if it is not the last one, bridge it
            wp1 = waypoints[-1]
            wp2 = ep_path[i+1]["path"][0]
            wps = [wp1, wp2]
            plan = xyplan_from_waypoints(wps, t, vel = 1, ep=None)
            full_plan += plan
            t += len(plan)
    return full_plan