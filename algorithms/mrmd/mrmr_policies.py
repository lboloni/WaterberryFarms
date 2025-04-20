"""
mrmr_policies.py

Functions helping to run experiments with the Waterberry Farms benchmark.

"""
from policy import AbstractCommunicateAndFollowPath
from communication import Message
from algorithms.mrmd.epmarket import EPM, EPAgent, EPOffer

class SimpleCommunicator(AbstractCommunicateAndFollowPath):
    """Example, simple communicator policy"""
    def __init__(self, exp_policy, exp_env):
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        super().__init__(vel, waypoints, repeat)
        self.name = exp_policy["policy-name"]
    
    def act(self, delta_t):
        """Call the following of the path"""
        super().act(delta_t)

    def act_send(self, round):
        print(f"{self.name} act_send called at round {round}")
        self.robot.com.send(self.robot, destination=None, message = Message("hello"))
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")

class MRMR_Pioneer(AbstractCommunicateAndFollowPath):
    """Implements the Pioneer agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        super().__init__(vel, waypoints, repeat)
        self.name = exp_policy["policy-name"]
        self.epagent = EPAgent(self.name)
        self.epm = EPM().epm
        print("MRMR_Pioneer policy was created")
    
    def act(self, delta_t):
        """Call the following of the path"""
        super().act(delta_t)

    def act_send(self, round):
        print(f"{self.name} act_send called at round {round}")
        self.robot.com.send(self.robot, destination=None, message = Message("hello"))
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")

class MRMR_Contractor(AbstractCommunicateAndFollowPath):
    """Implements the Contractor agent for the MRMR paper"""

    def __init__(self, exp_policy, exp_env):
        waypoints = [[5, 5 * exp_policy["seed"]]]
        repeat = False
        vel = 1
        super().__init__(vel, waypoints, repeat)
        self.name = exp_policy["policy-name"]
        self.epagent = EPAgent(self.name)
        self.epm = EPM().epm
        print("MRMR_Contractor policy was created")
    
    def act(self, delta_t):
        """Call the following of the path"""
        super().act(delta_t)

    def act_send(self, round):
        print(f"{self.name} act_send called at round {round}")
        self.robot.com.send(self.robot, destination=None, message = Message("hello"))
        
    def act_receive(self, round, messages):
        print(f"{self.name} act_receive called at round {round}")
        print(f"Messages {messages}")
