"""
communication.py

A model of inter-robot communication in the Waterberry Farms framework. 
"""
import copy

from robot import Robot


class Message:
    """Implements a message object with a free form content. Keeps track of sender, receiver, time sent and time received. """
    def __init__(self, content):
        self.content = content
        self.sender_name = None
        self.destination_name = None
        self.time_sent = -1
        self.time_received = -1

    def __repr__(self):
        return f"[Message: content:{self.content} sender_name:{self.sender_name} destination_name:{self.destination_name} time_sent:{self.time_sent} time_received:{self.time_received}]"


class CommunicationMedium:
    """A communication medium"""

    def __init__(self, env):
        """Creates a communication medium"""
        self.robots = {} # the robots in the environment
        self.mailboxes = {} # the mailboxes
        self.delivered_messages = [] # all the messages that had been delivered
        self.env = env
    
    def add_robot(self, robot):
        """Adds a robot to the system and creates the corresponding mailbox"""
        self.robots[robot.name] = robot
        self.mailboxes[robot.name] = []
        robot.com = self


class PerfectCommunicationMedium(CommunicationMedium):
    """A communication medium that covers the complete area where every message is delivered.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def send(self, sender: Robot, destination, message: Message):
        """A message is sent to a number of destinations."""
        for robot_name in self.mailboxes:
            if destination is None or destination==robot_name:
                if destination is None and robot_name==sender.name:
                    continue
                msg = copy.deepcopy(message)
                msg.sender_name = sender.name
                msg.destination_name = robot_name
                msg.time_sent = self.env.time
                self.mailboxes[robot_name].append(msg)

    def receive(self, receiver: Robot):
        """A robot picks up all the messages that were received"""
        retval = []
        mailbox = self.mailboxes[receiver.name]
        for msg in mailbox:
            msg.time_received = self.env.time
            retval.append(msg)
        mailbox.clear()
        return retval
        
