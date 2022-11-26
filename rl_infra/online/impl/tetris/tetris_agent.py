import random

from tetris.config import BOARD_SIZE
import torch

from ....offline.dqn import DeepQNetwork
from ...types.agent import Agent
from .tetris_environment import TetrisAction, TetrisState


class TetrisAgent(Agent[TetrisState, TetrisAction]):
    epsilon: float
    policy: DeepQNetwork

    def __init__(self, device: torch.device, epsilon: float = 0.1) -> None:
        self.epsilon = epsilon
        self.policy = DeepQNetwork(
            arrayHeight=BOARD_SIZE[0],
            arrayWidth=BOARD_SIZE[1],
            numOutputs=4,
            device=device,
        )

    def choosePolicyAction(self, state: TetrisState) -> TetrisAction:
        input = TetrisAgent.toDqnInput(state)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy(input).max(1)[1].view(1, 1)

    def chooseRandomAction(self) -> TetrisAction:
        return random.choice(
            [
                TetrisAction.DOWN,
                TetrisAction.LEFT,
                TetrisAction.RIGHT,
                TetrisAction.NONE,
            ]
        )

    @staticmethod
    def toDqnInput(state: TetrisState) -> torch.Tensor:
        # FIXME: This only uses the board, not score or current/next piece.
        return torch.from_numpy(state.board)
