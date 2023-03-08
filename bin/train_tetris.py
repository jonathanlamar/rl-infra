#!/usr/bin/env python3
import argparse

import torch

from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService
from rl_infra.impl.tetris.offline.tetris_training_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment
from rl_infra.types.offline.model_service import ModelDbKey


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
    print(f"Deployed version {modelDbKey.version} of model {modelDbKey.tag}")

    print("Loading deployed model")
    agent = TetrisAgent(device=device)
    print(
        f"Model loaded.  Agent has played {agent.numEpisodesPlayed} episodes and trained {agent.numEpochsTrained} "
        "epochs."
    )

    return agent


def playEpisodeWithRetraining(
    agent: TetrisAgent,
    env: TetrisEnvironment,
    args: argparse.Namespace,
) -> tuple[TetrisAgent, TetrisEnvironment]:
    gameIsOver = False
    while not gameIsOver:
        action = agent.chooseAction(env.currentState)
        transition = env.step(action)
        gameIsOver = transition.state.isTerminal
        print("Retraining model")
        trainingService.retrainAndPublish(
            modelDbKey=agent.dbKey,
            epochNumber=agent.numEpochsTrained,
            batchSize=args.batch_size,
            numBatches=args.num_batches,
        )
        agent = deployAndLoadModel(agent.dbKey)

    print(
        f"Episode {agent.numEpisodesPlayed} done. "
        f"Moves: {len(env.currentEpisodeRecord.moves)}, "
        f"Score: {env.currentState.score}. "
        f"Epsilon: {agent.epsilon:.4f}."
    )

    print("Saving episode")
    dataService.pushEpisode(env.currentEpisodeRecord)

    env.startNewEpisode()
    agent.startNewEpisode()

    return agent, env


def updateOnlineMetrics(modelDbKey: ModelDbKey, env: TetrisEnvironment, modelService: TetrisModelService) -> None:
    print("Updating online metrics for model")
    metrics = [ep.computeOnlineMetrics() for ep in env.currentGameplayRecord.episodes]
    avgEpisodeLength = sum([m.numMoves for m in metrics]) / len(metrics)
    avgScore = sum([m.score for m in metrics]) / len(metrics)
    print(f"Average episode length: {avgEpisodeLength:0.4f}.")
    print(f"Average score: {avgScore:0.4f}.")

    for m in metrics:
        modelService.publishOnlineMetrics(modelDbKey, m)


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataService = TetrisDataService()
    modelService = TetrisModelService()
    trainingService = TetrisTrainingService(device=device)

    modelDbKey = modelService.getModelKey(args.model_tag, args.version)
    agent = deployAndLoadModel(modelDbKey)
    env = TetrisEnvironment()

    for _ in range(args.num_episodes):
        agent, env = playEpisodeWithRetraining(agent, env, args)
        updateOnlineMetrics(agent.dbKey, env, modelService)

    print("Deleting old training examples")
    dataService.keepNewRowsDeleteOld(sgn=0, numToKeep=1000)
    dataService.keepNewRowsDeleteOld(sgn=-1, numToKeep=1000)
