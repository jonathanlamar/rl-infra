from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Type

import torch
from pydantic import validator
from tetris.config import BOARD_SIZE
from tetris.game import KeyPress
from tetris.utils import Tetramino
from torch import Tensor

from rl_infra.types.base_types import NumpyArray
from rl_infra.types.online.transition import Action, State, Transition


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
                return TetrisPiece(val.letter)
            except ValueError:
                raise TypeError(f"Unkown tetramino type {val}")
        elif isinstance(val, str):
            return TetrisPiece(val)
        raise TypeError(f"Invalid type for val: {type(val)}")


class TetrisState(State):
    board: NumpyArray[Literal["uint8"]]
    score: int
    activePiece: TetrisPiece
    nextPiece: TetrisPiece

    def toDqnInput(self) -> Tensor:
        return torch.from_numpy(self.board.copy().reshape(1, 2, BOARD_SIZE[0], BOARD_SIZE[1]))


class TetrisAction(Action, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NONE = "NONE"

    def toKeyPress(self) -> KeyPress:
        return KeyPress[self.value]


class TetrisTransition(Transition[TetrisState, TetrisAction]):
    @validator("state", "newState", pre=True)
    @classmethod
    def _parseStateFromJson(cls: Type[TetrisTransition], val: TetrisState | str) -> TetrisState:
        if isinstance(val, str):
            return TetrisState.parse_raw(val)
        return val
