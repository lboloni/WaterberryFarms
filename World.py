from Environment import Environment, DissipationModelEnvironment, EpidemicSpreadEnvironment
from InformationModel import InformationModel, GaussianProcessScalarFieldIM
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy

import logging
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.ERROR)

class World:
    def __init__(self, env, im):
        self.env = env
        self.im = im
        self.width, self.height = env.width, env.height
        self.time = 0
        self.robots = []
                
    def add_robot(self, robot):
        self.robots.append(robot)
        
    def enact_policy(self, delta_t = 1.0):
        """For all the agents in the environment, it calls 
        their policies to allow for the scheduling of the actions."""
        for robot in self.robots:
            robot.enact_policy(delta_t)
        
    def proceed(self, delta_t = 1.0):
        """Implements the passing of time"""
        for robot in self.robots:
            robot.proceed(delta_t)
        self.env.proceed(delta_t)
        self.im.proceed(delta_t)
        self.time = self.time + delta_t
        
class WorldFactory:
    @staticmethod
    def generate_environment_pretrained_static_pollution(width = 10, height = 10, seed = 1):
        """Generates a pollution model that had some polution but now it is static"""
        env = DissipationModelEnvironment("water", width, height, seed)
        env.evolve_speed = 1
        env.p_pollution = 0.1
        for i in range(100):
            env.proceed(1.0)
        # setting it to unchanging
        env.evolve_speed = 0
        env.p_pollution = 0.0
        return env
    
    @staticmethod
    def generate_im(env, settings):
        """Generates an information model appropriate to the environment and the estimation 
        type described by the settings"""
        ## FIXME: this should be different based on settings["estimation_type"]
        im = GaussianProcessScalarFieldIM("sample", 
                            env.width, env.height)
        return im

    @staticmethod
    def generate_world_pretrained_static_pollution(width = 10, height = 10, 
                                                   seed = 1, estimation_type = "gaussian-process"):
        env = WorldFactory.generate_environment_pretrained_static_pollution(width, height, seed)
        settings = {"estimation_type": estimation_type}
        im = WorldFactory.generate_im(env, settings)
        world = World(env, im)
        return world