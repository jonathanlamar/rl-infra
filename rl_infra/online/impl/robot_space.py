from dataclasses import dataclass
from enum import Enum

import numpy

from rl_infra.online.types import Action, ActionSpace, State, StateSpace
from rl_infra.online.utils import RobotClient


@dataclass
class RobotState(State):
    cameraImg: numpy.ndarray
    distanceSensor: int


class RobotStateSpace(StateSpace):
    def sample(self) -> RobotState:
        return RobotClient.getSensorReading()

    def contains(self, _: RobotState) -> bool:
        return False


RobotAction = Action


class RobotActionSpace(ActionSpace, Enum):
    MOVE_FORWARD = 1
    MOVE_BACKWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
