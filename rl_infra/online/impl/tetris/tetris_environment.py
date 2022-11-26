from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from numpy import ndarray
from tetris.game import Ell, Eye, GameState, KeyPress, Ohh, Tee, Zee
from tetris.utils import Tetramino

from ...types.environment import Action, Environment, State, Transition


class TetrisPiece(int, Enum):
    ELL = 0
    EYE = 1
    OHH = 2
    TEE = 3
    ZEE = 4

    @staticmethod
    def fromTetramino(tetramino: Tetramino) -> TetrisPiece:
        if isinstance(tetramino, Ell):
            return TetrisPiece.ELL
        elif isinstance(tetramino, Eye):
            return TetrisPiece.EYE
        elif isinstance(tetramino, Ohh):
            return TetrisPiece.OHH
        elif isinstance(tetramino, Tee):
            return TetrisPiece.TEE
        elif isinstance(tetramino, Zee):
            return TetrisPiece.ZEE
        else:
            raise KeyError(f"Unkown tetramino type {tetramino}")


@dataclass
class TetrisState(State):
    board: ndarray
    score: int
    currentPiece: TetrisPiece
    nextPiece: TetrisPiece

    @staticmethod
    def fromGameState(gameState: GameState) -> TetrisState:
        return TetrisState(
            board=gameState.board,
            score=gameState.score,
            currentPiece=TetrisPiece.fromTetramino(gameState.activePiece),
            nextPiece=TetrisPiece.fromTetramino(gameState.nextPiece),
        )


class TetrisAction(Action, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NONE = "NONE"

    def toKeyPress(self) -> KeyPress:
        if self == TetrisAction.UP:
            return KeyPress.UP
        elif self == TetrisAction.DOWN:
            return KeyPress.DOWN
        elif self == TetrisAction.LEFT:
            return KeyPress.LEFT
        elif self == TetrisAction.RIGHT:
            return KeyPress.RIGHT
        else:  # self == TetrisAction.NONE:
            return KeyPress.NONE


@dataclass
class TetrisTransition(Transition[TetrisState, TetrisAction]):
    state: TetrisState
    action: TetrisAction
    newState: TetrisState
    reward: float
    isTerminal: bool


class TetrisEnvironment(Environment[TetrisState, TetrisAction]):
    currentState: TetrisState
    gameState: GameState

    def __init__(self) -> None:
        self.gameState = GameState()
        self.currentState = TetrisState.fromGameState(self.gameState)

    def step(self, action: TetrisAction):
        self.gameState.update(action.toKeyPress())
        self.currentState = TetrisState.fromGameState(self.gameState)

    def getReward(
        self, oldState: TetrisState, action: TetrisAction, newState: TetrisState
    ) -> float:
        return newState.score - oldState.score
