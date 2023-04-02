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

logger = logging.getLogger(__name__)


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
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to play (default 10).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="""Number of transitions per batch (default 128).  Transitions will be selected at random from recent
        gameplay and distributed such that examples with positive, zero, and negative reward are roughly equal.""",
    )
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches per training epoch. (default 1)")

    return parser


def deployAndLoadModel(modelDbKey: ModelDbKey) -> TetrisAgent:
    modelService.deployModel(modelDbKey)
    return TetrisAgent(device=device)


def playEpisodeWithRetraining(
    agent: TetrisAgent,
    env: TetrisEnvironment,
    args: argparse.Namespace,
) -> tuple[TetrisAgent, TetrisEnvironment]:
    gameIsOver = False
    logging.info("Playing episode with retraining after each move.")
    while not gameIsOver:
        logging.debug(f"State: {env.currentState}")
        action = agent.chooseAction(env.currentState)
        logger.debug(f"Action: {action}")
        transition = env.step(action)
        gameIsOver = transition.state.isTerminal
        trainingService.retrainAndPublish(
            modelDbKey=agent.dbKey,
            epochNumber=agent.numEpochsTrained,
            batchSize=args.batch_size,
            numBatches=args.num_batches,
        )
        agent = deployAndLoadModel(agent.dbKey)

    logging.info(f"Episode played: {env.currentEpisodeRecord}")

    return agent, env


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    logging.info(f"args = {args}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"device = {device}")
    dataService = TetrisDataService()
    modelService = TetrisModelService()
    trainingService = TetrisTrainingService(device=device)

    modelDbKey = modelService.getModelKey(args.model_tag, args.version)
    logging.info(f"Deploying model {modelDbKey}.")
    agent = deployAndLoadModel(modelDbKey)
    env = TetrisEnvironment(episodeNumber=agent.numEpisodesPlayed)
    modelEntry = modelService.getModelEntry(modelDbKey)
    logging.info(f"Model entry retrieved: {modelEntry}.")

    for _ in range(args.num_episodes):
        assert modelEntry is not None, "Model entry should exist"
        logging.info(
            f"Epsilon: {agent.epsilon:.4f}\n"
            f"Num epochs trained: {modelEntry.numEpochsTrained:.4f}\n"
            f"Average episodes lenghth: {modelEntry.avgEpisodeLength or 0:.4f}\n"
            f"Recency weighted average loss: {modelEntry.recencyWeightedAvgLoss or 0:.4f}\n"
            f"Recency weighted validation average max Q: {modelEntry.recencyWeightedAvgValidationQ or 0:.4f}\n"
        )
        logging.info(f"Playing episode {modelEntry.numEpisodesPlayed}")
        agent, env = playEpisodeWithRetraining(agent, env, args)
        lastEpisode = env.currentEpisodeRecord
        onlineMetrics = lastEpisode.computeOnlineMetrics()
        env.startNewEpisode()
        agent.startNewEpisode()

        logging.info(
            f"Episodes played: {agent.numEpisodesPlayed}\n"
            f"Epochs trained: {agent.numEpochsTrained}\n"
            f"Moves: {onlineMetrics.numMoves}\n"
            f"Score: {onlineMetrics.score}\n"
        )

        logging.info("Saving episode")
        dataService.pushEpisode(env.currentEpisodeRecord)

        logging.info("Updating online metrics for model")
        modelService.publishOnlineMetrics(modelDbKey, onlineMetrics)
        modelEntry = modelService.getModelEntry(modelDbKey)

    logging.info("Deleting old training examples")
    dataService.keepNewRowsDeleteOld(sgn=0)
    dataService.keepNewRowsDeleteOld(sgn=-1)
