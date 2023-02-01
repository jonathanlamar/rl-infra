from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from ....base_types import NumpyArray, SerializableDataClass
from ...types.environment import Action, Environment, State, Transition
from .utils.robot_client import RobotClient, RobotSensorReading


@dataclass
class RobotState(State, RobotSensorReading, SerializableDataClass):
    image: NumpyArray[Literal["int64"]]
    distanceSweep: NumpyArray[Literal["int64"]]
    motionDetected: bool
    lightColorSensor: NumpyArray[Literal["int64"]]


class RobotAction(Action, Enum):
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    DO_NOTHING = "DO_NOTHING"


class RobotTransition(Transition[RobotState, RobotAction], SerializableDataClass):
    state: RobotState
    action: RobotAction
    newState: RobotState
    reward: float
    isTerminal: bool


class RobotEnvironment(ABC, Environment[RobotState, RobotAction]):
    currentState: RobotState
    moveStepSizeCm: int
    turnStepSizeDeg: int

    def __init__(self, moveStepSizeCm: int = 15, turnStepSizeDeg: int = 30) -> None:
        self.moveStepSizeCm = moveStepSizeCm
        self.turnStepSizeDeg = turnStepSizeDeg
        self.currentState = RobotEnvironment._getState()

    def step(self, action: RobotAction) -> RobotTransition:
        match action:
            case RobotAction.MOVE_FORWARD:
                RobotClient.sendAction("move", arg=self.moveStepSizeCm)
            case RobotAction.MOVE_BACKWARD:
                RobotClient.sendAction("move", arg=-self.moveStepSizeCm)
            case RobotAction.TURN_RIGHT:
                RobotClient.sendAction("turn", arg=self.turnStepSizeDeg)
            case RobotAction.TURN_LEFT:
                RobotClient.sendAction("turn", arg=-self.turnStepSizeDeg)
            case RobotAction.DO_NOTHING:
                pass
            case _:
                raise KeyError(f"Wrong action {action}")

        state = self.currentState
        newState = RobotEnvironment._getState()
        self.currentState = newState

        reward = self.getReward(self.currentState, action, newState)
        return RobotTransition(
            state=state,
            action=action,
            newState=newState,
            reward=reward,
            isTerminal=False,
        )

    @abstractmethod
    def getReward(self, oldState: RobotState, action: RobotAction, newState: RobotState) -> float:
        ...

    @staticmethod
    def _getState() -> RobotState:
        sensorReading = RobotClient.getSensorReading()
        return RobotState(**asdict(sensorReading))
