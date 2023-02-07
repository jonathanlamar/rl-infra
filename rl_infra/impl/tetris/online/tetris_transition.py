from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any, Literal, Type

import torch
from pydantic.utils import GetterDict
from tetris.config import BOARD_SIZE
from tetris.game import GameState, KeyPress
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


class TetrisGamestateGetterDict(GetterDict):
    """Special logic for parsing board of gamestate"""

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self._obj, GameState) and key == "board":
            board = deepcopy(self._obj.board)
            piece = self._obj.activePiece
            for idx in piece.squares:
                # Using 2 to encode meaning into the pixels of the active piece
                board[tuple(idx)] = 2
            return board
        else:
            return super().get(key, default)


class TetrisState(State):
    board: NumpyArray[Literal["uint8"]]
    score: int
    activePiece: TetrisPiece
    nextPiece: TetrisPiece

    def toDqnInput(self) -> Tensor:
        return torch.from_numpy(self.board.copy().reshape((1, 1) + BOARD_SIZE))

    class Config(State.Config):
        """pydantic config class"""

        orm_mode = True
        getter_dict = TetrisGamestateGetterDict


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
