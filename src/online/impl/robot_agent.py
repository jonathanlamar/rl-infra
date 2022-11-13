from typing import Protocol
from online.types import Agent, Policy
from online.utils import RobotDriver

class RobotAgent(Agent, Protocol):
    robot_driver: RobotDriver
    policy: Policy

    def __init__(self, policy: Policy, robot_driver: RobotDriver) -> None:
        self.robot_driver = robot_driver
        self.policy = policy
