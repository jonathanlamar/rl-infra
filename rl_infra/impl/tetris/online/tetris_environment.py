from __future__ import annotations

from time import time

from tetris.game import GameState
from tetris.utils.utils import KeyPress

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


class TetrisEpochRecord(EpochRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    def computeOnlineMetrics(self) -> TetrisOnlineMetrics:
        finalScore = max([move.newState.score for move in self.moves])
        return TetrisOnlineMetrics(avgEpochLength=len(self.moves), avgEpochScore=finalScore)


class TetrisGameplayRecord(GameplayRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    pass


class TetrisEnvironment(Environment[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    gameState: GameState
    humanPlayer: bool

    def __init__(self, humanPlayer: bool = False) -> None:
        self.humanPlayer = humanPlayer
        self.gameState = GameState()
        self.currentState = TetrisState.from_orm(self.gameState)
        self.currentGameplayRecord = TetrisGameplayRecord(epochs=[])
        self.currentEpochRecord = TetrisEpochRecord(moves=[])

    def step(self, action: TetrisAction) -> TetrisTransition:
        oldState = self.currentState
        self.gameState.update(action.toKeyPress())

        if (self.humanPlayer and time() - self.gameState.lastAdvanceTime > 0.25) or (not self.humanPlayer):
            self.gameState.update(KeyPress.DOWN)

        isTerminal = self.gameState.dead
        self.currentState = TetrisState.from_orm(self.gameState)
        reward = self.getReward(oldState, action, self.currentState)

        transition = TetrisTransition(
            state=oldState, action=action, newState=self.currentState, reward=reward, isTerminal=isTerminal
        )
        if transition.action != TetrisAction.NONE:
            self.currentEpochRecord = self.currentEpochRecord.append(transition)
        return transition

    def getReward(self, oldState: TetrisState, action: TetrisAction, newState: TetrisState) -> float:  # pyright: ignore
        if newState.isTerminal:
            return -10
        return newState.score - oldState.score

    def startNewEpoch(self) -> None:
        self.gameState = GameState()
        self.currentState = TetrisState.from_orm(self.gameState)
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpoch(self.currentEpochRecord)
        self.currentEpochRecord = TetrisEpochRecord(moves=[])
