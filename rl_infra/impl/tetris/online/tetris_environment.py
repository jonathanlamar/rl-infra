from __future__ import annotations

from tetris.game import GameState

from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState, TetrisTransition
from rl_infra.types.base_types import Metrics
from rl_infra.types.online.environment import Environment, EpochRecord, GameplayRecord


class TetrisOnlineMetrics(Metrics):
    avgEpochLength: float | None = None
    avgEpochScore: float | None = None

    def updateWithNewValues(self, other: TetrisOnlineMetrics) -> TetrisOnlineMetrics:
        """For each metric, compute average.  This results in an exponential recency weighted average."""

        return TetrisOnlineMetrics(
            avgEpochLength=TetrisOnlineMetrics.avgWithoutNone(self.avgEpochLength, other.avgEpochLength),
            avgEpochScore=TetrisOnlineMetrics.avgWithoutNone(self.avgEpochScore, other.avgEpochScore),
        )


class TetrisEpochRecord(EpochRecord[TetrisState, TetrisAction, TetrisTransition, TetrisOnlineMetrics]):
    epochNumber: int
    moves: list[TetrisTransition]

    def computeOnlineMetrics(self) -> TetrisOnlineMetrics:
        finalScore = max([move.newState.score for move in self.moves])
        return TetrisOnlineMetrics(avgEpochLength=len(self.moves), avgEpochScore=finalScore)


class TetrisGameplayRecord(GameplayRecord[TetrisState, TetrisAction, TetrisTransition, TetrisOnlineMetrics]):
    epochs: list[TetrisEpochRecord]


class TetrisEnvironment(Environment[TetrisState, TetrisAction, TetrisTransition, TetrisOnlineMetrics]):
    currentState: TetrisState
    currentEpochRecord: TetrisEpochRecord
    currentGameplayRecord: TetrisGameplayRecord
    gameState: GameState

    def __init__(self, epochNumber: int = 0) -> None:
        self.gameState = GameState()
        self.currentState = TetrisState.from_orm(self.gameState)
        self.currentGameplayRecord = TetrisGameplayRecord(epochs=[])
        self.currentEpochRecord = TetrisEpochRecord(epochNumber=epochNumber, moves=[])

    def step(self, action: TetrisAction) -> TetrisTransition:
        oldState = self.currentState
        self.gameState.update(action.toKeyPress())
        isTerminal = self.gameState.dead
        self.currentState = TetrisState.from_orm(self.gameState)
        reward = self.getReward(oldState, action, self.currentState)

        transition = TetrisTransition(
            state=oldState, action=action, newState=self.currentState, reward=reward, isTerminal=isTerminal
        )
        if transition.action != TetrisAction.NONE:
            self.currentEpochRecord = self.currentEpochRecord.append(transition)
        return transition

    def getReward(self, oldState: TetrisState, action: TetrisAction, newState: TetrisState) -> float:
        if newState.isTerminal:
            return -1
        return newState.score - oldState.score

    def startNewEpoch(self) -> None:
        self.gameState = GameState()
        self.currentState = TetrisState.from_orm(self.gameState)
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpoch(self.currentEpochRecord)
        self.currentEpochRecord = TetrisEpochRecord(epochNumber=self.currentEpochRecord.epochNumber + 1, moves=[])
