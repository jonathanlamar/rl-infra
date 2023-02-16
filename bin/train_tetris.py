#!/usr/bin/env python3
import argparse
from textwrap import dedent
from time import sleep

import torch
from IPython.terminal.embed import embed
from tetris.utils import KeyPress

from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService
from rl_infra.impl.tetris.offline.tetris_training_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment

NUM_BATCHES_PER_RETRAIN = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cold-start",
        action="store_true",
        help=dedent(
            """Whether to create new model weights from scratch.
            These will be stored as the latest version of the model tag used."""
        ),
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to play.  Retraining occurs every 10 batches by default.",
    )
    parser.add_argument(
        "-i",
        "--retrain-interval",
        type=int,
        default=10,
        help="How often (in epochs) to retrain the model.  Set to zero to skip retraining entirely.",
    )
    parser.add_argument(
        "-r", "--retrain-batches", type=int, default=10, help="Number of batches to retrain on per retraining session."
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help=dedent(
            """Whether to show agent gameplay with periodic calls to GameState.draw.  The
            value is the time delay in seconds between those calls.  Leave blank or set to
            zero to train without printing (much faster)"""
        ),
    )
    parser.add_argument(
        "-t",
        "--model-tag",
        type=str,
        default="throwaway",
        help="Which model tag to train.  Will always load the latest version unless --version is used.",
    )
    parser.add_argument(
        "-v", "--version", type=int, help="Model version to train.  Will default to latest version if blank."
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataService = TetrisDataService()
    modelService = TetrisModelService()
    trainingService = TetrisTrainingService(device=device)

    if args.cold_start:
        version = trainingService.coldStart(args.model_tag)
        print(f"Created version {version} of model {args.model_tag}")

    version = modelService.deployModel(args.model_tag, args.version)
    print(f"Deployed version {version} of model {args.model_tag}")

    print("Loading deployed model")
    agent = TetrisAgent(device=device)
    print(
        f"Model loaded.  Agent has played {agent.numEpochsPlayed} epochs.  "
        + f"Using epsilon-greedy choice with epsilon = {agent.epsilon}."
    )
    env = TetrisEnvironment()

    for _ in range(args.epochs):
        gameIsOver = False
        while not gameIsOver:
            action = agent.chooseAction(env.currentState)
            transition = env.step(action)
            env.gameState.update(KeyPress.DOWN)
            if args.print:
                env.gameState.draw()
                sleep(0.05)
            gameIsOver = transition.isTerminal
        print(
            f"Epoch {agent.numEpochsPlayed} done. There were {len(env.currentEpochRecord.moves)} moves, "
            f"and the final score was {env.currentState.score}."
        )
        env.startNewEpoch()
        agent.startNewEpoch()

        if args.retrain_interval != 0 and agent.numEpochsPlayed % args.retrain_interval == 0:
            print("Saving gameplay.")
            gameplay = env.currentGameplayRecord
            dataService.pushGameplay(gameplay)

            print("Retraining models")
            offlinePerformance = trainingService.retrainAndPublish(
                modelTag=agent.dbKey.tag, version=agent.dbKey.version, batchSize=128, numBatches=args.retrain_batches
            )

            print("Deleting old training examples")
            dataService.keepNewRowsDeleteOld(sgn=0, numToKeep=1000)
            dataService.keepNewRowsDeleteOld(sgn=-1, numToKeep=1000)

            print("Updating metrics for model")
            onlinePerformance = gameplay.computeOnlineMetrics()
            print(f"Online performance: {onlinePerformance}.  Offline performance: {offlinePerformance}")
            modelService.updateModel(
                agent.dbKey,
                numEpochsPlayed=args.retrain_interval,
                onlinePerformance=onlinePerformance,
                offlinePerformance=offlinePerformance,
            )

            version = modelService.deployModel(args.model_tag, args.version)
            print(f"Deployed version {version} of model {args.model_tag}")
            agent = TetrisAgent(device=device)
