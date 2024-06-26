#!/usr/bin/env python3

import argparse
import logging

import torch

from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService
from rl_infra.impl.tetris.offline.tetris_training_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment
from rl_infra.types.offline.model_service import ModelDbKey


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
    parser.add_argument(
        "--version", type=int, help="Model version to train.  Will default to the latest version if blank."
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to play (default 1).")
    parser.add_argument(
        "--retrain-interval",
        type=int,
        default=1,
        help="How often (in episodes) to retrain the model (default 1).  Set to zero to skip retraining during play.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="""Number of transitions per batch (default 32).  Transitions will be selected at random from recent
        gameplay and distributed such that examples with positive, zero, and negative reward are roughly equal.""",
    )
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches per training epoch. (default 1)")
    parser.add_argument(
        "--print",
        action="store_true",
        help="""Whether to show agent gameplay with periodic calls to GameState.draw.  Leave blank to train without
        printing (much faster)""",
    )

    return parser


def deployAndLoadModel(modelDbKey: ModelDbKey) -> TetrisAgent:
    modelService.deployModel(modelDbKey)
    return TetrisAgent(device=device)


def playEpisode(agent: TetrisAgent, env: TetrisEnvironment, logger: logging.Logger) -> TetrisEnvironment:
    gameIsOver = False
    while not gameIsOver:
        logger.debug(f"State: {env.currentState}")
        action = agent.chooseAction(env.currentState)
        logger.debug(f"Action: {action}")
        transition = env.step(action)
        gameIsOver = transition.newState.isTerminal
        logger.debug(f"Terminal: {gameIsOver}")

    return env


def retrainModel(agent: TetrisAgent, args: argparse.Namespace, trainingService: TetrisTrainingService) -> TetrisAgent:
    trainingService.retrainAndPublish(
        modelDbKey=agent.dbKey,
        epochNumber=agent.numEpochsTrained,
        batchSize=args.batch_size,
        numBatches=args.num_batches,
    )
    agent = deployAndLoadModel(agent.dbKey)

    return agent


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    logger = setupLogger()

    logger.info(f"args = {args}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")
    dataService = TetrisDataService()
    modelService = TetrisModelService()
    trainingService = TetrisTrainingService(device=device)

    modelDbKey = (
        modelService.getLatestVersionKey(args.model_tag)
        if args.version is None
        else modelService.getModelKey(args.model_tag, args.version)
    )
    if modelDbKey is None:
        raise RuntimeError("No model found.  Please run bin/cold_start_tetris.py")

    logger.info(f"Deploying model {modelDbKey}.")
    agent = deployAndLoadModel(modelDbKey)
    env = TetrisEnvironment(episodeNumber=agent.numEpisodesPlayed)
    modelEntry = modelService.getModelEntry(modelDbKey)
    logger.info(f"Model entry retrieved: {modelEntry}.")

    for _ in range(args.num_episodes):
        if modelEntry is None:
            raise RuntimeError("No model entry found.")
        logger.info(
            f"Epsilon: {agent.epsilon:.4f}\n"
            f"Num epochs trained: {modelEntry.numEpochsTrained:.4f}\n"
            f"Average episodes length: {modelEntry.avgEpisodeLength or 0:.4f}\n"
            f"Recency weighted average loss: {modelEntry.recencyWeightedAvgLoss or 0:.4f}\n"
            f"Recency weighted validation average max Q: {modelEntry.recencyWeightedAvgValidationQ or 0:.4f}\n"
        )

        env = playEpisode(agent, env, logger)
        lastEpisode = env.currentEpisodeRecord
        onlineMetrics = lastEpisode.computeOnlineMetrics()
        env.startNewEpisode()
        agent.startNewEpisode()

        logger.info(
            f"Episodes played: {agent.numEpisodesPlayed}\n"
            f"Epochs trained: {agent.numEpochsTrained}\n"
            f"Moves: {onlineMetrics.numMoves}\n"
            f"Score: {onlineMetrics.score}\n"
        )

        logger.info("Saving episode")
        dataService.pushEpisode(env.currentEpisodeRecord)

        logger.info("Updating online metrics for model")
        modelService.publishOnlineMetrics(modelDbKey, onlineMetrics)
        modelEntry = modelService.getModelEntry(modelDbKey)

        if args.retrain_interval != 0 and agent.numEpisodesPlayed % args.retrain_interval == 0:
            logger.info("Retraining model")
            agent = retrainModel(agent, args, logger, trainingService)

    logger.info("Deleting old training examples")
    dataService.keepNewRowsDeleteOld(sgn=0)
    dataService.keepNewRowsDeleteOld(sgn=-1)
