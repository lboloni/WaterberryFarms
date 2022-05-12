# unit tests for robot
# FIXME: for the time being these are just basic sanity tests, without being organized as unit tests

# allow the import from the source directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import Robot
from Environment import Environment, DissipationModelEnvironment, EpidemicSpreadEnvironment
from InformationModel import DiskEstimateScalarFieldIM
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy


env = DissipationModelEnvironment("water", 100, 100, seed=1)
env.evolve_speed = 1
env.p_pollution = 0.1
for t in range(90):
    env.proceed()
im = DiskEstimateScalarFieldIM("sample", env.width, env.height, disk_radius=10)