from __future__ import annotations

from abc import ABC, abstractmethod

from rl_infra.impl.robot.online.robot_client import RobotClient
from rl_infra.impl.robot.online.robot_transition import RobotAction, RobotState, RobotTransition
from rl_infra.types.online.environment import Environment, EpochRecord, GameplayRecord, OnlineMetrics


# TODO: Implement stubs here
class RobotOnlineMetrics(OnlineMetrics):
    def updateWithNewValues(self, other: RobotOnlineMetrics) -> RobotOnlineMetrics:
        return self


class RobotEpochRecord(EpochRecord[RobotState, RobotAction, RobotTransition, RobotOnlineMetrics]):
    epochNumber: int
    moves: list[RobotTransition]

    def computeOnlineMetrics(self) -> RobotOnlineMetrics:
        return RobotOnlineMetrics()


class RobotGameplayRecord(GameplayRecord[RobotState, RobotAction, RobotTransition, RobotOnlineMetrics]):
    epochs: list[RobotEpochRecord]


class RobotEnvironment(ABC, Environment[RobotState, RobotAction, RobotTransition, RobotOnlineMetrics]):
    currentState: RobotState
    moveStepSizeCm: int
    turnStepSizeDeg: int
    currentEpochRecord: RobotEpochRecord
    currentGameplayRecord: RobotGameplayRecord

    def __init__(self, epochNumber: int = 0, moveStepSizeCm: int = 15, turnStepSizeDeg: int = 30) -> None:
        self.moveStepSizeCm = moveStepSizeCm
        self.turnStepSizeDeg = turnStepSizeDeg
        self.currentState = RobotEnvironment._getState()
        self.currentEpochRecord = RobotEpochRecord(epochNumber=epochNumber, moves=[])
        self.currentGameplayRecord = RobotGameplayRecord(epochs=[])

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
        return RobotState(**sensorReading.dict())

    def startNewEpoch(self) -> None:
        self.currentState = RobotEnvironment._getState()
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpoch(self.currentEpochRecord)
        self.currentEpochRecord = RobotEpochRecord(epochNumber=self.currentEpochRecord.epochNumber + 1, moves=[])
