from __future__ import annotations

from copy import deepcopy
from time import time

import numpy as np
from numpy.typing import NDArray
from tetris.config.config import BOARD_SIZE
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
    stateBuffer: NDArray[np.uint8]

    def __init__(self, humanPlayer: bool = False) -> None:
        self.humanPlayer = humanPlayer
        self.gameState = GameState()
        self.stateBuffer = np.zeros((2, BOARD_SIZE[0], BOARD_SIZE[1] + 1), dtype=np.uint8)
        self.currentState = self._getCurrentState()
        self.currentGameplayRecord = TetrisGameplayRecord(epochs=[])
        self.currentEpochRecord = TetrisEpochRecord(moves=[])

    def _getCurrentState(self) -> TetrisState:
        return TetrisState(
            board=self.stateBuffer,  # pyright: ignore
            score=self.gameState.score,
            activePiece=self.gameState.activePiece,  # pyright: ignore
            nextPiece=self.gameState.nextPiece,  # pyright: ignore
            isTerminal=self.gameState.dead,
        )

    def step(self, action: TetrisAction) -> TetrisTransition:
        oldState = self.currentState
        self.gameState.update(action.toKeyPress())

        if not self.gameState.dead and (
            (self.humanPlayer and time() - self.gameState.lastAdvanceTime > 0.25) or (not self.humanPlayer)
        ):
            self.gameState.update(KeyPress.DOWN)

        self._updateBuffer()
        self.currentState = self._getCurrentState()
        reward = self.getReward(oldState, action, self.currentState)

        transition = TetrisTransition(
            state=oldState,
            action=action,
            newState=self.currentState,
            reward=reward,
        )
        self.currentEpochRecord = self.currentEpochRecord.append(transition)
        return transition

    def _updateBuffer(self) -> None:
        board = np.concatenate(
            [
                deepcopy(self.gameState.board).reshape((1,) + BOARD_SIZE),
                np.zeros((1, BOARD_SIZE[0], 1), dtype=np.uint8),
            ],
            axis=2,
        )
        for idx in self.gameState.activePiece.squares:
            board[0, idx[0], idx[1]] = 2
        board[0, 0, -1] = ["I", "L", "O", "T", "Z"].index(self.gameState.nextPiece.letter)
        board[0, 1, -1] = int(self.gameState.dead)
        self.stateBuffer = np.concatenate([board, self.stateBuffer[:-1, :, :]])

    def getReward(self, oldState: TetrisState, action: TetrisAction, newState: TetrisState) -> float:  # pyright: ignore
        if newState.isTerminal:
            return -1
        return newState.score - oldState.score

    def startNewEpoch(self) -> None:
        self.gameState = GameState()
        self.currentState = TetrisState.from_orm(self.gameState)
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpoch(self.currentEpochRecord)
        self.currentEpochRecord = TetrisEpochRecord(moves=[])
