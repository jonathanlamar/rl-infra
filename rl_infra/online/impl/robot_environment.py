from dataclasses import dataclass
from enum import Enum

import numpy

from ..types.environment import Action, Environment, State, StepOutcome
from ..utils.robot_client import RobotClient


@dataclass
class RobotState(State):
    cameraImg: numpy.ndarray
    distanceSensor: int


@dataclass
class RobotStepOutcome(StepOutcome[RobotState]):
    newState: RobotState
    reward: float
    isTerminal: bool


RobotAction = Action


class RobotActionSpace(str, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"


class RobotEnvironment(Environment[RobotState, RobotAction]):
    currentState: RobotState
    moveStepSizeCm: int
    turnStepSizeDeg: int

    def __init__(self, moveStepSizeCm: int = 1, turnStepSizeDeg: int = 15) -> None:
        self.moveStepSizeCm = moveStepSizeCm
        self.turnStepSizeDeg = turnStepSizeDeg
        self.currentState = RobotEnvironment._getState()

    def step(self, action: RobotAction) -> StepOutcome:
        if action == RobotActionSpace.MOVE_FORWARD:
            RobotClient.sendAction(1, arg=self.moveStepSizeCm)
        elif action == RobotActionSpace.MOVE_BACKWARD:
            RobotClient.sendAction(1, arg=-self.moveStepSizeCm)
        elif action == RobotActionSpace.TURN_RIGHT:
            RobotClient.sendAction(2, arg=self.turnStepSizeDeg)
        elif action == RobotActionSpace.TURN_LEFT:
            RobotClient.sendAction(2, arg=-self.turnStepSizeDeg)
        else:
            raise KeyError(f"Wrong action {action}")

        newState = RobotEnvironment._getState()
        self.currentState = newState

        # TODO: Where does the reward logic live?
        return RobotStepOutcome(newState=newState, reward=0, isTerminal=False)

    @staticmethod
    def _getState() -> RobotState:
        img, dist = RobotClient.getSensorReading()
        return RobotState(cameraImg=img, distanceSensor=dist)
