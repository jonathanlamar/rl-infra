#!/usr/bin/env python3
import argparse
import logging
import random

import torch

from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_training_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment, TetrisEpisodeRecord
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction

logger = logging.getLogger(__name__)


def getParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-tag",
        type=str,
        default="throwaway",
        help="Which model tag to train.  Will always load the latest version unless --version is used.",
    )

    return parser


def generateRandomEpisode() -> TetrisEpisodeRecord:
    env = TetrisEnvironment()
    gameIsOver = False
    logger.debug("Generating random episode")
    while not gameIsOver:
        logger.debug(f"State: {env.currentEpisodeRecord}")
        action = random.choice(list(TetrisAction))
        logger.debug(f"Action: {action}")
        transition = env.step(action)
        gameIsOver = transition.state.isTerminal

    logger.info(f"Generated episode {env.currentEpisodeRecord}")
    return env.currentEpisodeRecord  # pyright: ignore


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    logger.info(f"args = {args}")

    dataService = TetrisDataService()
    # The training loop samples before any on-policy data has been saved, so we seed with a random episode.
    trainEpisode = generateRandomEpisode()
    dataService.pushEpisode(trainEpisode)
    # We hold out a separate random episode for validation using average max-Q as a qualitative performance metric.
    valEpisode = generateRandomEpisode()
    dataService.pushValidationEpisode(valEpisode)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")
    trainingService = TetrisTrainingService(device)
    trainingService.coldStart(args.model_tag)
