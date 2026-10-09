"""
communication.py

A model of inter-robot communication in the Waterberry Farms framework. 
"""
import copy
import json

from robot import Robot


def message_size(content):
    """The size of a message content in bytes: the length of its compact JSON encoding. NumPy scalars
    are encoded as their Python values; content that is not plain data raises a TypeError."""
    def plain(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Message content is not plain data: {value!r}")
    return len(json.dumps(content, separators=(",", ":"), default=plain).encode("utf-8"))


def traffic_summary(traffic, timesteps):
    """Plain-data totals of the traffic records of a medium over a run of the given number of timesteps.
    Transmitted bytes count every send once, delivered bytes once per recipient."""
    summary = {"messages": 0, "bytes-transmitted": 0, "bytes-delivered": 0,
               "per-type": {}, "per-sender": {}, "bytes-per-timestep": [0] * int(timesteps)}
    for record in traffic:
        delivered = record["bytes"] * record["recipients"]
        summary["messages"] += 1
        summary["bytes-transmitted"] += record["bytes"]
        summary["bytes-delivered"] += delivered
        per_type = summary["per-type"].setdefault(record["type"],
            {"messages": 0, "bytes-transmitted": 0, "bytes-delivered": 0})
        per_type["messages"] += 1
        per_type["bytes-transmitted"] += record["bytes"]
        per_type["bytes-delivered"] += delivered
        per_sender = summary["per-sender"].setdefault(record["sender"], {"messages": 0, "bytes-transmitted": 0})
        per_sender["messages"] += 1
        per_sender["bytes-transmitted"] += record["bytes"]
        summary["bytes-per-timestep"][record["timestep"]] += record["bytes"]
    series = summary["bytes-per-timestep"]
    summary["mean-bytes-per-timestep"] = sum(series) / len(series) if series else 0
    summary["max-bytes-per-timestep"] = max(series) if series else 0
    return summary


class Message:
    """Implements a message object. Keeps track of sender, receiver, time sent and time received, the 
    times being (timestep, round) pairs. By convention, the content is a dict with a "type" key and plain 
    data values: every recipient receives a deep copy, so shared objects must be identified by ids. """
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
        self.traffic = [] # one record per send, for measuring the bandwidth (traffic_summary)
        self.env = env
        # the current (timestep, round), set by simulate_1day before every communication round
        self.timestep = None
        self.round = None

    def record_send(self, sender, destination, message, recipients):
        """Record a send in the traffic, with the number of robots it was delivered to"""
        content = message.content
        self.traffic.append({"timestep": self.timestep, "round": self.round, "sender": sender.name,
            "destination": destination, "type": content.get("type") if isinstance(content, dict) else None,
            "bytes": message_size(content), "recipients": recipients})

    def add_robot(self, robot):
        """Adds a robot to the system and creates the corresponding mailbox"""
        if robot.name in self.robots:
            raise Exception(f"Robot name {robot.name} is already registered")
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
        recipients = 0
        for robot_name in self.mailboxes:
            if destination is None or destination==robot_name:
                if destination is None and robot_name==sender.name:
                    continue
                msg = copy.deepcopy(message)
                msg.sender_name = sender.name
                msg.destination_name = robot_name
                msg.time_sent = (self.timestep, self.round)
                self.mailboxes[robot_name].append(msg)
                recipients += 1
        self.record_send(sender, destination, message, recipients)

    def receive(self, receiver: Robot):
        """A robot picks up all the messages that were received"""
        if self.robots[receiver.name] is not receiver:
            raise Exception(f"Mailbox {receiver.name} is not owned by this robot")
        retval = []
        mailbox = self.mailboxes[receiver.name]
        for msg in mailbox:
            msg.time_received = (self.timestep, self.round)
            retval.append(msg)
        mailbox.clear()
        return retval
        
