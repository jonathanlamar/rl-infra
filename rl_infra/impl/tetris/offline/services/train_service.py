from typing import Any

import torch
from tetris.config import BOARD_SIZE

from rl_infra.impl.tetris.offline.models.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.services.data_service import TetrisDataService
from rl_infra.impl.tetris.offline.services.model_service import TetrisModelService
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.types.offline.model_service import ModelDbKey, ModelType


class TetrisTrainingService:
    modelService: TetrisModelService
    dataService: TetrisDataService
    device: torch.device
    modelInitArgs: dict[str, Any]

    def __init__(self, device: torch.device) -> None:
        self.modelService = TetrisModelService()
        self.dataService = TetrisDataService()
        self.device = device
        self.modelInitArgs = {
            "arrayHeight": BOARD_SIZE[0],
            "arrayWidth": BOARD_SIZE[1],
            "numOutputs": 5,
            "device": self.device,
        }

    def modelFactory(self) -> DeepQNetwork:
        return DeepQNetwork(**self.modelInitArgs)

    def coldStart(self, modelTag: str):
        self.modelService.publishNewModel(
            model=self.modelFactory(),
            key=ModelDbKey(modelType=ModelType.ACTOR, modelTag=modelTag),
        )
        self.modelService.publishNewModel(
            model=self.modelFactory(),
            key=ModelDbKey(modelType=ModelType.CRITIC, modelTag=modelTag),
        )

    def retrainAndPublish(self, modelTag: str, batchSize: int, numBatches):
        actorEntry = self.modelService.getModelEntry(modelTag, ModelType.ACTOR)
        actor = self.modelFactory()
        actor.load_state_dict(torch.load(actorEntry.modelLocation))
        criticEntry = self.modelService.getModelEntry(modelTag, ModelType.CRITIC)
        critic = self.modelFactory()
        critic.load_state_dict(torch.load(criticEntry.modelLocation))

        batch = self.dataService.sample(batchSize)
        nonFinalMask = torch.tensor(
            tuple(map(lambda s: not s.transition.isTerminal, batch)), device=self.device, dtype=torch.bool
        )
        nonFinalNextStates = torch.cat(
            [elt.transition.newState.toDqnInput() for elt in batch if not elt.transition.isTerminal]
        )

        stateBatch = torch.cat([elt.transition.state.toDqnInput() for elt in batch])
        actionBatch = torch.tensor([TetrisAgent.possibleActions.index(elt.transition.action) for elt in batch])
        rewardBatch = torch.tensor([elt.transition.reward for elt in batch])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        stateActionValues = actor(stateBatch).gather(1, actionBatch.reshape(-1, 1)).reshape(-1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        nextStateValues = torch.zeros(batchSize, device=self.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = critic(nonFinalNextStates).max(1)[0]

        # Compute expected Q values
        GAMMA = 0.99
        expectedStateActionValues = (nextStateValues * GAMMA) + rewardBatch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(stateActionValues, expectedStateActionValues)

        # Optimize the model
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, amsgrad=True)
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(actor.parameters(), 100)
        optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        TAU = 0.005
        criticStateDict = critic.state_dict()
        actorStateDict = actor.state_dict()
        for key in actorStateDict:
            criticStateDict[key] = actorStateDict[key] * TAU + criticStateDict[key] * (1 - TAU)
        critic.load_state_dict(criticStateDict)

        self.modelService.updateModel(actorEntry.modelDbKey, model=actor, numBatchesTrained=numBatches)
        self.modelService.updateModel(criticEntry.modelDbKey, model=critic, numBatchesTrained=numBatches)
