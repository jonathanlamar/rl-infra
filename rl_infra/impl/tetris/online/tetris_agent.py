import random

import torch
from tetris.config import BOARD_SIZE

from rl_infra.impl.tetris.offline.models.dqn import DeepQNetwork
from rl_infra.impl.tetris.online.config import MODEL_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisAction, TetrisState
from rl_infra.types.online.agent import Agent


class TetrisAgent(Agent[TetrisState, TetrisAction]):
    epsilon: float
    policy: DeepQNetwork

    def __init__(self, device: torch.device, epsilon: float = 0.1, coldStart: bool = False) -> None:
        self.epsilon = epsilon
        self.policy = DeepQNetwork(
            arrayHeight=BOARD_SIZE[0],
            arrayWidth=BOARD_SIZE[1],
            numOutputs=5,
            device=device,
        )
        if not coldStart:
            self.policy.load_state_dict(torch.load(MODEL_ROOT_PATH))
        # Make sure the models always see the same order
        self._possibleActions = list(sorted(TetrisAction))

    def choosePolicyAction(self, state: TetrisState) -> TetrisAction:
        input = TetrisAgent.toDqnInput(state)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            prediction = self.policy(input).max(1)[1].view(1).numpy()[0]
        return self._possibleActions[prediction]

    def chooseRandomAction(self) -> TetrisAction:
        return random.choice(self._possibleActions)

    @staticmethod
    def toDqnInput(state: TetrisState) -> torch.Tensor:
        return torch.from_numpy(state.board.reshape((1, 1) + BOARD_SIZE))
