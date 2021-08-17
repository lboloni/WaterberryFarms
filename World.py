from Environment import Environment, PollutionModelEnvironment, EpidemicSpreadEnvironment
from InformationModel import InformationModel, ScalarFieldInformationModel_stored_observation
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
        
def generate_environment():
    """Function to generate the environment model"""
    env = PollutionModelEnvironment("water", 10, 10, seed = 1)
    env.evolve_speed = 1
    env.p_pollution = 0.1
    for i in range(100):
        env.proceed(1.0)
    # setting it to unchanging
    env.evolve_speed = 0
    env.p_pollution = 0.0
    return env

def generate_information_model(settings, env):
    """Generates an information model of a particular type"""
    im = ScalarFieldInformationModel_stored_observation("sample", 
                        env.width, env.height, 
                        estimation_type=settings["estimation_type"])
    return im

def generate_robots(settings, world):
    """Generate the robots and their policies according to the settings. 
    Current version generate n robots with random waypoint behavior"""
    count = settings["count"]
    for i in range(count):
        robot = Robot(f"Robot-{i}", 0, 0, 0, env=world.env, im=world.im)
        robot.policy = RandomWaypointPolicy(None, robot, 1, [0,0], [9, 9], seed = 1)
        world.add_robot(robot)
        
def run_experiment(genenv, genim, genrob):
    """Generates the environment, the model and the robots"""
    env = genenv()
    im = genim(env)
    world = World(env, im)
    genrob(world)
    record = []
    for t in range(100):
        world.enact_policy(1)
        world.proceed(1)
        ## this is where we get some kind of metric
        record.append(im.score(env))
    return record