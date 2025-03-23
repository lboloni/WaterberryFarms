# unit tests for robot
# FIXME: for the time being these are just basic sanity tests, without being organized as unit tests

# allow the import from the source directory
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Robot import Robot
from environment import Environment, DissipationModelEnvironment, EpidemicSpreadEnvironment
from InformationModel import DiskEstimateScalarFieldIM
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy


env = DissipationModelEnvironment("water", 100, 100, seed=1)
env.evolve_speed = 1
env.p_pollution = 0.1
for t in range(90):
    env.proceed()
im = DiskEstimateScalarFieldIM("sample", env.width, env.height, disk_radius=10)

robot = Robot("Robi", 20, 30, 0)
robot.env = env
robot.im = im
# print(vars(robot))
print(robot)
robot.add_action("North")
robot.enact_policy()
robot.proceed()
print(robot)

robot.x = math.pi
print(robot.toHTML())

robot.policy = FollowPathPolicy(None, robot, 1, [[0,0], [5, 5], [9,0]], repeat = True)


value = robot.toHTML()
value += "<br>Policy: "
if robot.policy == None:
    value += "None"
else: 
    value += str(robot.policy)
value += "<br>Pending actions:"
value += str(robot.pending_actions)