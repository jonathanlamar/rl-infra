from dataclasses import dataclass
from enum import Enum
from typing import Any

from online.types import StateSpace, ActionSpace, State, Action
from online.utils import RobotDriver


@dataclass
class RobotState(State):
    # TODO: This should be a dataclass containing a snapshot of all sensor
    # readings at a time.  That would be readings from the camera and the
    # distance sensor.  I don't think the camera has a microphone.
    cameraImg: Any
    distanceSensor: Any


class RobotStateSpace(StateSpace):
    def __init__(self, robot_driver: RobotDriver) -> None:
        self.robot_driver = robot_driver

    def sample(self) -> RobotState:
        return self.robot_driver.takeSensorReading()

    def contains(self, _: RobotState) -> bool:
        return False


RobotAction = Action


class RobotActionSpace(ActionSpace, Enum):
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
