#!/usr/bin/env python3
import torch

from rl_infra.offline.tetris.services.data_service import DataDbEntry, DataService
from rl_infra.offline.tetris.services.model_service import ModelDbKey, ModelService, ModelType
from rl_infra.online.impl.tetris.config import MODEL_ROOT_PATH
from rl_infra.online.impl.tetris.tetris_agent import TetrisAgent
from rl_infra.online.impl.tetris.tetris_environment import TetrisEnvironment

dataService = DataService()
modelService = ModelService()
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
