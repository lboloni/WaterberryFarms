import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pprint

from communication import PerfectCommunicationMedium, Message
from environment import ScalarFieldEnvironment
from robot import Robot

class TestPerfectCommunicationMedium(unittest.TestCase):

    def test_message_delivery(self):
        # self.assertEqual(2+3, 5)
        env = ScalarFieldEnvironment("noname", 100, 100, seed=1)
        com = PerfectCommunicationMedium(env)
        robot1 = Robot("robot-1", init_x=0, init_y=0, init_altitude=1, env=env)
        com.add_robot(robot1)
        robot2 = Robot("robot-2", init_x=0, init_y=0, init_altitude=1, env=env)
        com.add_robot(robot2)
        robot3 = Robot("robot-3", init_x=0, init_y=0, init_altitude=1, env=env)
        com.add_robot(robot3)
        com.send(robot1, destination=None, message = Message("hello"))
        # pick up the messages
        msgs1 = com.receive(robot1)
        pprint.pprint(msgs1)
        msgs2 = com.receive(robot2)
        pprint.pprint(msgs2)
        msgs3 = com.receive(robot3)
        pprint.pprint(msgs3)


if __name__ == '__main__':
    unittest.main()