#!/usr/bin/env python3
import torch

from rl_infra.impl.tetris.offline.services.data_service import TetrisDataService
from rl_infra.impl.tetris.offline.services.model_service import TetrisModelService
from rl_infra.impl.tetris.offline.services.train_service import TetrisTrainingService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment, TetrisEpoch, TetrisGameplayRecord
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

for _ in range(10):
    epochs = []
    for _ in range(10):
        numMovesPlayed = 0
        gameIsOver = False
        transitions = []
        env = TetrisEnvironment()
        while not gameIsOver:
            action = agent.chooseAction(env.currentState)
            transition = env.step(action)
            transitions.append(transition)
            numMovesPlayed += 1
            gameIsOver = transition.isTerminal
        print(f"Epoch {epochIndex} done.  There were {numMovesPlayed} moves.")
        epoch = TetrisEpoch(epochNumber=epochIndex, moves=transitions)
        epochs.append(epoch)
        epochIndex += 1

    gameplay = TetrisGameplayRecord(epochs=epochs)

    print("Saving gameplay.")
    dataService.pushGameplay(gameplay)

    print("Updating online metrics for model")
    modelService.updateModel(
        ModelDbKey(modelTag=modelTag, modelType=ModelType.ACTOR),
        numEpochsPlayed=10,
        onlinePerformance=gameplay.computeOnlineMetrics(),
    )

    print("Retraining models")
    trainingService.retrainAndPublish(modelTag, batchSize=32, numBatches=10)
