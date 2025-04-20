# FIXME: this is an experiment for the communicating agents, it 
# does not belong here

class SimpleCommunicator(AbstractCommunicateAndFollowPath):
    def __init__(self, exp_policy, exp_env):
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        super().__init__(vel, waypoints, repeat)
        name = exp_policy["policy-name"]
    
    def act_send(self, round):
        print(f"{self.name} act_send called at round {round}")
        self.robot.com.send(self.robot, destination=None, message = Message("hello"))
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")

# FIXME: this is an experiment for the communicating agents, it 
# does not belong here

def generateLocalCommunicator(exp_policy, exp_env):
    """Example of how to generate a policy with a local specification. For the time being, we are creating a random waypoint one..."""
    #
    #  Random waypoint policy
    #
    #print("Generate Local Communicator with spec {exp_policy}")
    #geo = get_geometry(exp_env["typename"])
    #policy = RandomWaypointPolicy(
    #     vel = 1, low_point = [geo["xmin"], 
    #     geo["ymin"]], high_point = [geo["xmax"], geo["ymax"]], seed = exp_policy["seed"])  
    # policy.name = exp_policy["policy-name"]
    policy = SimpleCommunicator(exp_policy, exp_env)
    return policy
