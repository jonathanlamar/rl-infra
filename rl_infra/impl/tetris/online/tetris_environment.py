from __future__ import annotations

from tetris.game import GameState, KeyPress

from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState, TetrisTransition
from rl_infra.types.online.environment import Environment, EpochRecord, GameplayRecord, OnlineMetrics


class TetrisOnlineMetrics(OnlineMetrics):
    avgEpochLength: float | None = None
    avgEpochScore: float | None = None

    def updateWithNewValues(self, other: TetrisOnlineMetrics) -> TetrisOnlineMetrics:
        """For each metric, compute average.  This results in an exponential recency weighted average."""

        def avgWithoutNone(num1: float | None, num2: float | None) -> float | None:
            nums = [x for x in [num1, num2] if x is not None]
            if not nums:
                return None
            return sum(nums) / len(nums)

        return TetrisOnlineMetrics(
            avgEpochLength=avgWithoutNone(self.avgEpochLength, other.avgEpochLength),
            avgEpochScore=avgWithoutNone(self.avgEpochScore, other.avgEpochScore),
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
        self.gameState.update(KeyPress.NONE)  # To prevent model exploiting a bug in the game
        isTerminal = self.gameState.dead
        self.currentState = TetrisState.from_orm(self.gameState)
        reward = self.getReward(oldState, action, self.currentState)

        transition = TetrisTransition(
            state=oldState, action=action, newState=self.currentState, reward=reward, isTerminal=isTerminal
        )
        self.currentEpochRecord = self.currentEpochRecord.append(transition)
        return transition

    def getReward(self, oldState: TetrisState, action: TetrisAction, newState: TetrisState) -> float:
        return 1 + newState.score - oldState.score

    def startNewEpoch(self) -> None:
        self.gameState = GameState()
        self.currentState = TetrisState.from_orm(self.gameState)
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpoch(self.currentEpochRecord)
        self.currentEpochRecord = TetrisEpochRecord(epochNumber=self.currentEpochRecord.epochNumber + 1, moves=[])
