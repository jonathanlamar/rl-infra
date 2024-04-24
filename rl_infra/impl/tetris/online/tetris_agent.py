import logging
import math
import random

import torch
from tetris.config import BOARD_SIZE

from rl_infra.impl.tetris.offline.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.tetris_schema import TetrisModelDbEntry
from rl_infra.impl.tetris.online.config import (
    EPSILON_DECAY_RATE,
    FINAL_EPSILON,
    INITIAL_EPSILON,
    MODEL_ENTRY_PATH,
    MODEL_WEIGHTS_PATH,
)
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState
from rl_infra.types.online.agent import Agent

logger = logging.getLogger(__name__)


class TetrisAgent(Agent[TetrisState, TetrisAction, DeepQNetwork]):
    possibleActions = list(sorted(TetrisAction))  # Make sure the models always see the same order

    def __init__(self, device: torch.device) -> None:
        self.policy = DeepQNetwork(
            arrayHeight=BOARD_SIZE[0],
            arrayWidth=BOARD_SIZE[1] + 1,
            numOutputs=5,
            device=device,
        )
        self.policy.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
        entry = TetrisModelDbEntry.parse_file(MODEL_ENTRY_PATH)
        self.dbKey = entry.modelDbKey
        self.numEpisodesPlayed = entry.numEpisodesPlayed
        self.numEpochsTrained = entry.numEpochsTrained
        self.epsilon = self._updateEpsilon()

    def startNewEpisode(self) -> None:
        self.numEpisodesPlayed += 1
        self.epsilon = self._updateEpsilon()
        logger.debug(f"Starting new episode.  epsilon = {self.epsilon}")

    def choosePolicyAction(self, state: TetrisState) -> TetrisAction:
        logger.debug("Choosing policy action")
        input = state.toDqnInput()
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            prediction: int = self.policy(input).max(1)[1].view(1).numpy()[0]
        return self.possibleActions[prediction]

    def chooseRandomAction(self) -> TetrisAction:
        logger.debug("Choosing random action")
        return random.choice(self.possibleActions)

    def _updateEpsilon(self) -> float:
        return FINAL_EPSILON + (INITIAL_EPSILON - FINAL_EPSILON) * math.exp(
            -1.0 * self.numEpisodesPlayed / EPSILON_DECAY_RATE
        )
