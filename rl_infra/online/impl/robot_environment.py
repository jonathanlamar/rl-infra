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


class RobotAction(Action, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    DO_NOTHING = "DO_NOTHING"


class RobotEnvironment(Environment[RobotState, RobotAction]):
    currentState: RobotState
    moveStepSizeCm: int
    turnStepSizeDeg: int

    def __init__(self, moveStepSizeCm: int = 5, turnStepSizeDeg: int = 20) -> None:
        self.moveStepSizeCm = moveStepSizeCm
        self.turnStepSizeDeg = turnStepSizeDeg
        self.currentState = RobotEnvironment._getState()

    def step(self, action: RobotAction) -> StepOutcome:
        if action == RobotAction.MOVE_FORWARD:
            RobotClient.sendAction("move", arg=self.moveStepSizeCm)
        elif action == RobotAction.MOVE_BACKWARD:
            RobotClient.sendAction("move", arg=-self.moveStepSizeCm)
        elif action == RobotAction.TURN_RIGHT:
            RobotClient.sendAction("turn", arg=self.turnStepSizeDeg)
        elif action == RobotAction.TURN_LEFT:
            RobotClient.sendAction("turn", arg=-self.turnStepSizeDeg)
        elif action != RobotAction.DO_NOTHING:
            raise KeyError(f"Wrong action {action}")

        newState = RobotEnvironment._getState()
        self.currentState = newState

        # TODO: Where does the reward logic live?
        return RobotStepOutcome(newState=newState, reward=0, isTerminal=False)

    @staticmethod
    def _getState() -> RobotState:
        img, dist = RobotClient.getSensorReading()
        return RobotState(cameraImg=img, distanceSensor=dist)
