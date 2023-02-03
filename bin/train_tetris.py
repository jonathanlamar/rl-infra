#!/usr/bin/env python3
import torch

from rl_infra.impl.tetris.offline.services.data_service import DataDbEntry, TetrisDataService
from rl_infra.impl.tetris.offline.services.model_service import ModelDbKey, TetrisModelService, ModelType
from rl_infra.impl.tetris.online.config import MODEL_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_environment import TetrisEnvironment

dataService = TetrisDataService()
modelService = TetrisModelService()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
agent = TetrisAgent(device, coldStart=True)
torch.save(agent.policy.state_dict(), MODEL_ROOT_PATH)
modelTag = "jonExperimentOne"  # Some username tag

for epoch in range(10):
    env = TetrisEnvironment()
    move = 0
    gameIsOver = False
    transitions = []
    while not gameIsOver:
        action = agent.chooseAction(env.currentState)
        transition = env.step(action)
        transitions.append(DataDbEntry(transition=transition, epoch=epoch, move=move))
        move += 1
        gameIsOver = transition.isTerminal
    print(f"Epoch {epoch} done.  There were {move} total moves.  Saving data...")
    dataService.push(transitions)

# TODO: Train model on collected data

modelService.publishModel(model=agent.policy, modelDbKey=ModelDbKey(modelType=ModelType.ACTOR, modelTag=modelTag))
modelService.publishModel(model=agent.policy, modelDbKey=ModelDbKey(modelType=ModelType.CRITIC, modelTag=modelTag))
