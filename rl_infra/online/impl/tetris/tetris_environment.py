from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from tetris.game import Ell, Eye, GameState, KeyPress, Ohh, Tee, Zee
from tetris.utils import Tetramino

from ....base_types import NumpyArray, SerializableDataClass
from ...types.environment import Action, Environment, State, Transition


class TetrisPiece(int, Enum):
    ELL = 0
    EYE = 1
    OHH = 2
    TEE = 3
    ZEE = 4

    @staticmethod
    def fromTetramino(tetramino: Tetramino) -> TetrisPiece:
        match tetramino:
            case Ell():
                return TetrisPiece.ELL
            case Eye():
                return TetrisPiece.EYE
            case Ohh():
                return TetrisPiece.OHH
            case Tee():
                return TetrisPiece.TEE
            case Zee():
                return TetrisPiece.ZEE
            case _:
                raise KeyError(f"Unkown tetramino type {tetramino}")


@dataclass
class TetrisState(State, SerializableDataClass):
    board: NumpyArray[Literal["int64"]]
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
        match self:
            case TetrisAction.UP:
                return KeyPress.UP
            case TetrisAction.DOWN:
                return KeyPress.DOWN
            case TetrisAction.LEFT:
                return KeyPress.LEFT
            case TetrisAction.RIGHT:
                return KeyPress.RIGHT
            case _:
                return KeyPress.NONE


@dataclass
class TetrisTransition(Transition[TetrisState, TetrisAction], SerializableDataClass):
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

    def step(self, action: TetrisAction) -> TetrisTransition:
        oldState = self.currentState
        self.gameState.update(action.toKeyPress())
        isTerminal = self.gameState.dead
        self.currentState = TetrisState.fromGameState(self.gameState)
        reward = self.getReward(oldState, action, self.currentState)

        return TetrisTransition(
            state=oldState, action=action, newState=self.currentState, reward=reward, isTerminal=isTerminal
        )

    def getReward(self, oldState: TetrisState, action: TetrisAction, newState: TetrisState) -> float:
        return newState.score - oldState.score
