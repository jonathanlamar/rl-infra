#!/usr/bin/env python3
import json

import torch

from rl_infra.impl.tetris.offline.services.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.services.tetris_model_service import TetrisModelService, TetrisOfflineMetrics
from rl_infra.impl.tetris.offline.services.tetris_training_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment
from rl_infra.types.offline.model_service import ModelDbKey

EPOCHS_PER_RETRAIN = 10
NUM_RETRAINS = 10
NUM_BATCHES_PER_RETRAIN = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataService = TetrisDataService()
modelService = TetrisModelService()
trainingService = TetrisTrainingService(device=device)

modelTag = "jonalarmEndToEndTest"
trainingService.coldStart(modelTag)
modelService.deployModel()
agent = TetrisAgent(device=device)
epochIndex = agent.numEpochsPlayed
env = TetrisEnvironment(epochNumber=epochIndex)

allLosses = []
for _ in range(NUM_RETRAINS):
    for _ in range(EPOCHS_PER_RETRAIN):
        gameIsOver = False
        while not gameIsOver:
            action = agent.chooseAction(env.currentState)
            transition = env.step(action)
            gameIsOver = transition.isTerminal
        print(
            f"Epoch {env.currentEpochRecord.epochNumber} done. "
            + f"There were {len(env.currentEpochRecord.moves)} moves, and the final score was {env.currentState.score}."
        )
        env.startNewEpoch()

    print("Saving gameplay.")
    gameplay = env.currentGameplayRecord
    dataService.pushGameplay(gameplay)

    print("Retraining models")
    losses = trainingService.retrainAndPublish(
        modelTag=modelTag, version=0, batchSize=128, numBatches=NUM_BATCHES_PER_RETRAIN
    )
    allLosses += losses

    print("Updating metrics for model")
    onlinePerformance = gameplay.computeOnlineMetrics()
    offlinePerformance = TetrisOfflineMetrics.fromList(losses)
    print(f"Online performance: {onlinePerformance}.  Offline performance: {offlinePerformance}")
    modelService.updateModel(
        ModelDbKey(tag=modelTag, version=0),
        numEpochsPlayed=EPOCHS_PER_RETRAIN,
        onlinePerformance=onlinePerformance,
        offlinePerformance=offlinePerformance,
    )

    with open("notebook/allLosses.json", "w") as f:
        json.dump(allLosses, f)
