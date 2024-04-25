#!/usr/bin/env python3

import argparse
import logging
import random

import torch

from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_training_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment, TetrisEpisodeRecord
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction


def setupLogger() -> logging.Logger:
    logger = logging.getLogger("rl_infra")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logger.level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


def getParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-tag",
        type=str,
        default="throwaway",
        help="Which model tag to train.  Will always load the latest version unless --version is used.",
    )

    return parser


def generateRandomEpisode(logger: logging.Logger) -> TetrisEpisodeRecord:
    env = TetrisEnvironment()
    gameIsOver = False
    logger.debug("Generating random episode")
    while not gameIsOver:
        logger.debug(f"State: {env.currentEpisodeRecord}")
        action = random.choice(list(TetrisAction))
        logger.debug(f"Action: {action}")
        transition = env.step(action)
        gameIsOver = transition.state.isTerminal

    logger.info(
        f"Generated episode {env.currentEpisodeRecord.episodeNumber} with {len(env.currentEpisodeRecord.moves)} moves."
    )
    logger.debug(f"Episode moves: {env.currentEpisodeRecord.moves}")
    return env.currentEpisodeRecord  # pyright: ignore


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    logger = setupLogger()

    logger.info(f"args = {args}")

    dataService = TetrisDataService()
    # The training loop samples before any on-policy data has been saved, so we seed with a random episode.
    trainEpisode = generateRandomEpisode(logger)
    dataService.pushEpisode(trainEpisode)
    # We hold out a separate random episode for validation using average max-Q as a qualitative performance metric.
    valEpisode = generateRandomEpisode(logger)
    dataService.pushValidationEpisode(valEpisode)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")
    trainingService = TetrisTrainingService(device)
    trainingService.coldStart(args.model_tag)
