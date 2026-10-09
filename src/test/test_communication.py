import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from communication import Message, PerfectCommunicationMedium, message_size, traffic_summary
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
        self.medium.timestep, self.medium.round = 4, 1
        self.medium.send(self.robots[0], None, Message({"type": "hello"}))
        self.assertEqual(self.medium.receive(self.robots[0]), [])
        for receiver in self.robots[1:]:
            messages = self.medium.receive(receiver)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0].content, {"type": "hello"})
            self.assertEqual(messages[0].sender_name, "robot-0")
            self.assertEqual(messages[0].destination_name, receiver.name)
            self.assertEqual(messages[0].time_sent, (4, 1))
            self.assertEqual(messages[0].time_received, (4, 1))
            self.assertEqual(self.medium.receive(receiver), [])

    def test_directed_message_has_one_recipient(self):
        self.medium.send(self.robots[0], "robot-2", Message("direct"))
        self.assertEqual(self.medium.receive(self.robots[1]), [])
        self.assertEqual(self.medium.receive(self.robots[2])[0].content, "direct")

    def test_duplicate_names_and_foreign_mailbox_access_fail(self):
        with self.assertRaisesRegex(Exception, "already registered"):
            self.medium.add_robot(Robot("robot-0", 0, 0, 0))
        with self.assertRaisesRegex(Exception, "not owned"):
            self.medium.receive(Robot("robot-1", 0, 0, 0))


class TestTraffic(unittest.TestCase):
    def setUp(self):
        self.medium = PerfectCommunicationMedium(ScalarFieldEnvironment("field", 3, 3, seed=0))
        self.robots = [Robot(f"robot-{index}", 0, 0, 0) for index in range(3)]
        for robot in self.robots:
            self.medium.add_robot(robot)

    def test_message_size_is_the_compact_json_length(self):
        self.assertEqual(message_size({"type": "hello"}), len('{"type":"hello"}'))
        self.assertEqual(message_size({"x": np.int64(3), "v": np.float64(0.5)}), len('{"x":3,"v":0.5}'))
        with self.assertRaises(TypeError):
            message_size({"robot": self.robots[0]})

    def test_every_send_is_recorded_with_its_recipients(self):
        self.medium.timestep, self.medium.round = 2, 1
        self.medium.send(self.robots[0], None, Message({"type": "hello"}))
        self.medium.send(self.robots[1], "robot-2", Message("direct"))
        self.assertEqual(self.medium.traffic, [
            {"timestep": 2, "round": 1, "sender": "robot-0", "destination": None, "type": "hello",
             "bytes": 16, "recipients": 2},
            {"timestep": 2, "round": 1, "sender": "robot-1", "destination": "robot-2", "type": None,
             "bytes": 8, "recipients": 1}])

    def test_summary_counts_transmitted_and_delivered_bytes(self):
        for timestep in [0, 2, 2]:
            self.medium.timestep, self.medium.round = timestep, 0
            self.medium.send(self.robots[0], None, Message({"type": "hello"}))
        summary = traffic_summary(self.medium.traffic, 4)
        self.assertEqual((summary["messages"], summary["bytes-transmitted"], summary["bytes-delivered"]),
                         (3, 48, 96))
        self.assertEqual(summary["per-type"], {"hello": {"messages": 3, "bytes-transmitted": 48, "bytes-delivered": 96}})
        self.assertEqual(summary["per-sender"], {"robot-0": {"messages": 3, "bytes-transmitted": 48}})
        self.assertEqual(summary["bytes-per-timestep"], [16, 0, 32, 0])
        self.assertEqual((summary["mean-bytes-per-timestep"], summary["max-bytes-per-timestep"]), (12, 32))

    def test_no_traffic_gives_zero_totals(self):
        summary = traffic_summary([], 3)
        self.assertEqual((summary["messages"], summary["bytes-transmitted"], summary["bytes-per-timestep"],
                          summary["max-bytes-per-timestep"]), (0, 0, [0, 0, 0], 0))


if __name__ == "__main__":
    unittest.main()
