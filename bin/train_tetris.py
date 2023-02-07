#!/usr/bin/env python3
import torch

from rl_infra.impl.tetris.offline.services.data_service import TetrisDataService
from rl_infra.impl.tetris.offline.services.model_service import TetrisModelService
from rl_infra.impl.tetris.offline.services.train_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment
from rl_infra.types.offline.model_service import ModelDbKey, ModelType

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
for _ in range(100):
    for _ in range(10):
        gameIsOver = False
        while not gameIsOver:
            action = agent.chooseAction(env.currentState)
            transition = env.step(action)
            gameIsOver = transition.isTerminal
        print(
            f"Epoch {env.currentEpochRecord.epochNumber} done.  There were {len(env.currentEpochRecord.moves)} moves."
        )
        env.startNewEpoch()

    print("Saving gameplay.")
    gameplay = env.currentGameplayRecord
    dataService.pushGameplay(gameplay)

    print("Updating online metrics for model")
    modelService.updateModel(
        ModelDbKey(modelTag=modelTag, modelType=ModelType.ACTOR),
        numEpochsPlayed=10,
        onlinePerformance=gameplay.computeOnlineMetrics(),
    )

    print("Retraining models")
    trainingService.retrainAndPublish(modelTag, batchSize=128, numBatches=1)
