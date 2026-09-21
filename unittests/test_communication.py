import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from communication import Message, PerfectCommunicationMedium
from environment import ScalarFieldEnvironment
from robot import Robot


class TestPerfectCommunicationMedium(unittest.TestCase):
    def setUp(self):
        self.environment = ScalarFieldEnvironment("field", 3, 3, seed=0)
        self.medium = PerfectCommunicationMedium(self.environment)
        self.robots = [Robot(f"robot-{index}", 0, 0, 0) for index in range(3)]
        for robot in self.robots:
            self.medium.add_robot(robot)

    def test_broadcast_excludes_sender_and_clears_mailbox(self):
        self.environment.time = 4
        self.medium.send(self.robots[0], None, Message("hello"))
        self.assertEqual(self.medium.receive(self.robots[0]), [])
        for receiver in self.robots[1:]:
            messages = self.medium.receive(receiver)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].content, "hello")
            self.assertEqual(messages[0].sender_name, "robot-0")
            self.assertEqual(messages[0].destination_name, receiver.name)
            self.assertEqual(messages[0].time_sent, 4)
            self.assertEqual(messages[0].time_received, 4)
            self.assertEqual(self.medium.receive(receiver), [])

    def test_directed_message_has_one_recipient(self):
        self.medium.send(self.robots[0], "robot-2", Message("direct"))
        self.assertEqual(self.medium.receive(self.robots[1]), [])
        self.assertEqual(self.medium.receive(self.robots[2])[0].content, "direct")


if __name__ == "__main__":
    unittest.main()
