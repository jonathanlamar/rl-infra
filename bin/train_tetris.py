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


def getParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--cold-start",
        action="store_true",
        help=dedent(
            """
            Whether to create new model weights from scratch.
            These will be stored as the latest version of the model tag used.
            """
        ),
    )
    parser.add_argument(
        "-t",
        "--model-tag",
        type=str,
        default="throwaway",
        help=dedent(
            """
            Which model tag to train.  Will always load the latest version unless --version
            is used.
            """
        ),
    )
    parser.add_argument(
        "-v", "--version", type=int, help="Model version to train.  Will default to the latest version if blank."
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=10,
        help=dedent(
            """
            Number of epochs to play (default 10).  Retraining occurs every RETRAIN_INTERVAL
            batches.
            """
        ),
    )
    parser.add_argument(
        "-i",
        "--retrain-interval",
        type=int,
        default=10,
        help=dedent(
            """
            How often (in epochs) to retrain the model (default 10).  Set to zero to skip
            retraining entirely.
            """
        ),
    )
    parser.add_argument(
        "-r", "--retrain-batches", type=int, default=10, help="Number of batches per retraining session (default 10)."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help=dedent(
            """
            Number of transitions per batch (default 128).  Transitions will be selected at
            random from recent gameplay and distributed such that examples with positive,
            zero, and negative reward are roughly equal.
            """
        ),
    )
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help=dedent(
            """
            Whether to show agent gameplay with periodic calls to GameState.draw.  Leave
            blank to train without printing (much faster)
            """
        ),
    )
    parser.add_argument(
        "-g",
        "--greedy-policy",
        action="store_true",
        help=dedent(
            """
            Whether to use a greedy policy in action selection which always chooses the
            action with highest value. Leave blank to use an epsilon-greedy policy with
            exponentially decaying epsilon as a function of the number of epochs played.
            """
        ),
    )

    return parser


def deployAndLoadModel(args: argparse.Namespace) -> TetrisAgent:
    version = modelService.deployModel(args.model_tag, args.version)
    print(f"Deployed version {version} of model {args.model_tag}")

    print("Loading deployed model")
    agent = TetrisAgent(device=device)
    print(f"Model loaded.  Agent has played {agent.numEpochsPlayed} epochs.  ")

    return agent


def playEpoch(
    agent: TetrisAgent, env: TetrisEnvironment, args: argparse.Namespace
) -> tuple[TetrisAgent, TetrisEnvironment]:
    gameIsOver = False
    if args.print:
        # Give the user a chance to read logs before the screen is cleared.
        sleep(5)
    while not gameIsOver:
        action = agent.chooseAction(env.currentState)
        transition = env.step(action)
        env.gameState.update(KeyPress.DOWN)
        if args.print:
            env.gameState.draw()
            sleep(0.05)
        gameIsOver = transition.isTerminal
    print(
        f"Epoch {agent.numEpochsPlayed} done. "
        f"Moves: {len(env.currentEpochRecord.moves)}, "
        f"Score: {env.currentState.score}. "
        f"Epsilon: {agent.epsilon:.4f}."
    )
    env.startNewEpoch()
    agent.startNewEpoch()

    return agent, env


def updateGamePlayData(
    env: TetrisEnvironment, dataService: TetrisDataService, modelService: TetrisModelService, args: argparse.Namespace
) -> None:
    print("Saving gameplay.")
    gameplay = env.currentGameplayRecord
    dataService.pushGameplay(gameplay)

    print("Deleting old training examples")
    dataService.keepNewRowsDeleteOld(sgn=0, numToKeep=1000)
    dataService.keepNewRowsDeleteOld(sgn=-1, numToKeep=1000)

    print("Updating metrics for model")
    onlinePerformance = gameplay.computeOnlineMetrics()
    print(f"Average epoch length: {onlinePerformance.avgEpochLength:0.4f}.")
    print(f"Average score: {onlinePerformance.avgEpochScore:0.4f}.")

    modelService.updateModel(
        agent.dbKey,
        numEpochsPlayed=args.retrain_interval,
        onlinePerformance=onlinePerformance,
    )


def retrainModelAndUpdateMetrics(agent: TetrisAgent, args: argparse.Namespace) -> None:
    print("Retraining models")
    offlinePerformance = trainingService.retrainAndPublish(
        modelTag=agent.dbKey.tag,
        version=agent.dbKey.version,
        batchSize=args.batch_size,
        numBatches=args.retrain_batches,
    )
    print(f"Average training loss: {offlinePerformance.avgTrainingLoss:0.4f}")

    modelService.updateModel(
        agent.dbKey,
        offlinePerformance=offlinePerformance,
    )


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataService = TetrisDataService()
    modelService = TetrisModelService()
    trainingService = TetrisTrainingService(device=device)

    if args.cold_start:
        version = trainingService.coldStart(args.model_tag)
        print(f"Created version {version} of model {args.model_tag}")

    agent = deployAndLoadModel(args)
    env = TetrisEnvironment()

    for _ in range(args.num_epochs):
        playEpoch(agent, env, args)

        if args.retrain_interval != 0 and agent.numEpochsPlayed % args.retrain_interval == 0:
            updateGamePlayData(env, dataService, modelService, args)
            retrainModelAndUpdateMetrics(agent, args)
            agent = deployAndLoadModel(args)
