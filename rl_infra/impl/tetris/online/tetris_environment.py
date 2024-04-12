from __future__ import annotations

import logging
from copy import deepcopy
from time import time

import numpy as np
from numpy.typing import NDArray
from tetris.config.config import BOARD_SIZE
from tetris.game import GameState
from tetris.utils.utils import KeyPress

from rl_infra.impl.tetris.offline.tetris_schema import TetrisOnlineMetrics
from rl_infra.impl.tetris.online.tetris_transition import (
    TetrisAction,
    TetrisState,
    TetrisStateActionSequence,
    TetrisTransition,
)
from rl_infra.types.online.environment import Environment, EpisodeRecord, GameplayRecord

logger = logging.getLogger(__name__)


class TetrisEpisodeRecord(EpisodeRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    def computeOnlineMetrics(self) -> TetrisOnlineMetrics:
        return TetrisOnlineMetrics(
            episodeNumber=self.episodeNumber,
            numMoves=len(self.moves),
            score=max([move.newState.score for move in self.moves]),
        )


TetrisGameplayRecord = GameplayRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]


class TetrisEnvironment(Environment[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    gameState: GameState
    humanPlayer: bool
    stateBuffer: NDArray[np.uint8]

    def __init__(self, episodeNumber: int = 0, humanPlayer: bool = False) -> None:
        self.humanPlayer = humanPlayer
        self.gameState = GameState()
        self.stateBuffer = np.zeros((2, BOARD_SIZE[0], BOARD_SIZE[1] + 1), dtype=np.uint8)
        self.currentState = self._getCurrentState()
        self.currentGameplayRecord = TetrisGameplayRecord(episodes=[])
        self.currentEpisodeRecord = TetrisEpisodeRecord(episodeNumber=episodeNumber, moves=[])

    def _getCurrentState(self) -> TetrisState:
        return TetrisState(
            board=self.stateBuffer,  # pyright: ignore
            score=self.gameState.score,
            activePiece=self.gameState.activePiece,  # pyright: ignore
            nextPiece=self.gameState.nextPiece,  # pyright: ignore
            isTerminal=self.gameState.dead,
        )

    def step(self, action: TetrisAction) -> TetrisTransition:
        logger.debug("Stepping environment")
        oldState = self.currentState
        self.gameState.update(action.toKeyPress())

        if not self.gameState.dead and (
            (self.humanPlayer and time() - self.gameState.lastAdvanceTime > 0.25) or (not self.humanPlayer)
        ):
            self.gameState.update(KeyPress.DOWN)

        self._updateBuffer()
        self.currentState = self._getCurrentState()
        stateActionSequence = TetrisStateActionSequence(states=[oldState, self.currentState], actions=[action])
        reward = self.getReward(stateActionSequence)

        transition = TetrisTransition(
            state=oldState,
            action=action,
            newState=self.currentState,
            reward=reward,
        )
        logger.debug(f"transition = {transition}")
        self.currentEpisodeRecord = self.currentEpisodeRecord.append(transition)
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

    def getReward(self, stateActionSequence: TetrisStateActionSequence) -> float:  # pyright: ignore
        if stateActionSequence.states[1].isTerminal:
            return -1
        return stateActionSequence.states[1].score - stateActionSequence.states[0].score

    def startNewEpisode(self) -> None:
        logger.info("Staring new episode.")
        self.gameState = GameState()
        self.currentState = self._getCurrentState()
        self.currentGameplayRecord = self.currentGameplayRecord.appendEpisode(self.currentEpisodeRecord)
        self.currentEpisodeRecord = TetrisEpisodeRecord(
            episodeNumber=self.currentEpisodeRecord.episodeNumber + 1, moves=[]
        )
        logger.debug(f"Gameplay record = {self.currentGameplayRecord}")
