import ast
from collections import deque

class Robot:
    """The representation of a robot / drone.
    
    A class to gather the various functionalities of a robot (or drone) performing informative path planning. 

    The robot has a current location including a height. It proceeds through a series of actions. Some actions (in the list ```everystep_actions```) are executed always, while the ```pending_actions``` list is the one which is added by the policy. Actions include movement action, actions that directly set the location, velocity and acceleration, as well as actions for observation. There are some simplified actions like ```north''', ```west``` etc.

    The execution of the robot proceeds through timesteps, which are at a time delta_t from the previous timestep. At each timestep there are two phases:

    * ```enact_policy```: the policies associated with the robot are called and have the ability to schedule actions for execution 
    * ```proceed```: actions are executed. The actions (in the list ```everystep_actions```) are executed always, while the ```pending_actions``` list is the one which is added by the policy, and are cleared at this step.
    """    
    
    def __init__(self, name, init_x, init_y, init_altitude, grid_resolution=1, env=None, im=None):
        self.name = name
        self.init_x, self.init_y, self.init_altitude = init_x, init_y, init_altitude        
        self.x, self.y, self.altitude = init_x, init_y, init_altitude
        self.vel_x = self.vel_y = self.vel_altitude = 0
        # actions requested at a particular timestep
        self.pending_actions = []
        if env == None:
            self.everystep_actions = ["Move", "History"]
        else:
            self.everystep_actions = ["Move", "Observe", "History"]
        self.grid_resolution = grid_resolution
        self.energy = 100 # energy level
        self.value = 0 # accumulated value
        self.location_history = deque()
        self.policy = None
        self.env = env
        self.im = im
    
    def assign_policy(self, policy):
        """Assign a policy to the robot and set up the links from both directions"""
        self.policy = policy
        policy.robot = self

    def add_action(self, action):
        """Sets a pending action: normally this means to move. """
        self.pending_actions.append(action)

        
    def enact_policy(self, delta_t = 1.0):
        """Call the policy, if any to schedule actions"""
        if self.policy != None:
            self.policy.act(delta_t)
        
        
    def proceed(self, delta_t = 1.0):
        """Enacts all the pending and everystep actions.
        Updates the energy and value accumulated"""
        for action in self.pending_actions:
                self.enact_action(action, delta_t)
        self.pending_actions = []
        for action in self.everystep_actions:
                self.enact_action(action, delta_t)

    def add_observation(self, obs):
        """This allows the robot to also take into account the observation it makes. We are going to pass it to the policy. Most, but not all policies will ignore it."""
        self.policy.add_observation(obs)

        
    def enact_action(self, action, delta_t = 1.0):
        """Enacts one pending action. We are allowing here for a couple of shorthand
        actions like east, west, south, north..."""
        if action == "Observe":
            # simple point observation, skip it if we are outside the environment
            try:
                reading = self.env.value[int(self.x),int(self.y)]
            except IndexError:
                return
            obs = {"x": self.x, "y": self.y, "value": reading}
            self.im.add_observation(obs)
            self.energy = self.energy - 1
            # FIXME: this should be the VoI
            self.value = self.value + 1
            return
        if action == "Move":
            self.x = self.x + delta_t * self.vel_x
            self.y = self.y + delta_t * self.vel_y
            self.altitude = self.altitude + delta_t * self.vel_altitude
            return
        if action == "History": # appends x, y, altitude 
            self.location_history.appendleft([self.x, self.y, self.altitude])
            return
        if action == "West":
            self.vel_x = - self.grid_resolution
            self.vel_y = 0
            self.vel_altitude = 0
            return
        if action == "East":
            self.vel_x = self.grid_resolution
            self.vel_y = 0
            self.vel_altitude = 0
            return
        if action == "North": # this is the opposite of what one would think due to coord system
            self.vel_x = 0
            self.vel_y = -self.grid_resolution
            self.vel_altitude = 0
            return
        if action == "South": # this is the opposite of what one would think due to coord system
            self.vel_x = 0
            self.vel_y = self.grid_resolution
            self.vel_altitude = 0
            return
        if action[0:4] == "loc ":
            params = ast.literal_eval(action[4:])
            self.x = params[0]
            self.y = params[1]
            if len(params) > 2:
                self.altitude = params[2]
            return
        if action[0:4] == "vel ":
            params = ast.literal_eval(action[4:])
            self.vel_x = params[0]
            self.vel_y = params[1]
            if len(params) > 2:
                self.vel_altitude = params[2]
            return
        if action[0:4] == "acc ":
            params = ast.literal_eval(action[4:])
            self.vel_x = self.vel_x + delta_t * params[0]
            self.vel_y = self_vel_y + delta_t * params[1]
            if len(params) > 2:
                self.vel_altitude = self.vel_altitude + delta_t * params[2]
            return    
        
        ## FIXME: add an action for ascend, descend, observe
        raise Exception(f"Unsupported action {action} for robot {self.name}")
        
        
    def toHTML(self):
        """Simple HTML formatting"""
        value = f"<b>{self.name}</b><br/> loc = [x:{self.x:.2f},y:{self.y:.2f}, alt:{self.altitude:.2f}]<br/>" + \
            f"vel = [x:{self.vel_x:.2f},y:{self.vel_y:.2f},alt:{self.vel_altitude:.2f}]" 
        value += "<br>Policy: "
        if self.policy == None:
            value += "None"
        else: 
            value += str(self.policy)
        value += "<br>Pending actions:"
        value += str(self.pending_actions)
        return value
            
    def __str__(self):
        """Simple text formatting"""
        value = f"{self.name} --> loc = [x:{self.x:.2f},y:{self.y:.2f},alt:{self.altitude:.2f}] " + \
            f"vel = [x:{self.vel_x:.2f},y:{self.vel_y:.2f},alt:{self.vel_altitude:.2f}]" 
        return value