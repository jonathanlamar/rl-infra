from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Type

from tetris.game import GameState, KeyPress
from tetris.utils import Tetramino

from rl_infra.base_types import NumpyArray
from rl_infra.online.types.environment import Action, Environment, State, Transition


class TetrisPiece(str, Enum):
    ELL = "L"
    EYE = "I"
    OHH = "O"
    TEE = "T"
    ZEE = "Z"

    @classmethod
    def __get_validators__(cls: Type[TetrisPiece]):
        yield cls.validators

    @classmethod
    def validators(cls: Type[TetrisPiece], val: Any) -> TetrisPiece:
        if isinstance(val, Tetramino):
            try:
                TetrisPiece(val.letter)
            except ValueError:
                raise TypeError(f"Unkown tetramino type {val}")
        raise TypeError(f"Invalid type for val: {type(val)}")


class TetrisState(State):
    board: NumpyArray[Literal["int64"]]
    score: int
    activePiece: TetrisPiece
    nextPiece: TetrisPiece


class TetrisAction(Action, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NONE = "NONE"

    def toKeyPress(self) -> KeyPress:
        return KeyPress[self.value]


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
        self.currentState = TetrisState.from_orm(self.gameState)

    def step(self, action: TetrisAction) -> TetrisTransition:
        oldState = self.currentState
        self.gameState.update(action.toKeyPress())
        isTerminal = self.gameState.dead
        self.currentState = TetrisState.from_orm(self.gameState)
        reward = self.getReward(oldState, action, self.currentState)

        return TetrisTransition(
            state=oldState, action=action, newState=self.currentState, reward=reward, isTerminal=isTerminal
        )

    def getReward(self, oldState: TetrisState, action: TetrisAction, newState: TetrisState) -> float:
        return 1 + newState.score - oldState.score
