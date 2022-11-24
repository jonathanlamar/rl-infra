from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from numpy import ndarray

from ...types.environment import Action, Environment, State, StepOutcome
from ...utils.robot_client import RobotClient


@dataclass
class RobotState(State):
    cameraImg: ndarray
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


class RobotEnvironment(ABC, Environment[RobotState, RobotAction]):
    currentState: RobotState
    moveStepSizeCm: int
    turnStepSizeDeg: int

    def __init__(self, moveStepSizeCm: int = 15, turnStepSizeDeg: int = 30) -> None:
        self.moveStepSizeCm = moveStepSizeCm
        self.turnStepSizeDeg = turnStepSizeDeg
        self.currentState = RobotEnvironment._getState()

    def step(self, action: RobotAction) -> RobotStepOutcome:
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

        reward = self.getReward(self.currentState, action, newState)
        return RobotStepOutcome(newState=newState, reward=reward, isTerminal=False)

    @abstractmethod
    def getReward(
        self, oldState: RobotState, action: RobotAction, newState: RobotState
    ) -> float:
        ...

    @staticmethod
    def _getState() -> RobotState:
        img, dist = RobotClient.getSensorReading()
        return RobotState(cameraImg=img, distanceSensor=dist)
