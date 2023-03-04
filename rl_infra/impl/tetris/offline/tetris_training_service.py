from typing import Any, Sequence

import torch
from tetris.config import BOARD_SIZE
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import AdamW, Optimizer

from rl_infra.impl.tetris.offline.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService, TetrisOfflineMetrics
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState
from rl_infra.types.offline.training_service import TrainingService
from rl_infra.types.online.transition import Transition

FUTURE_REWARDS_DISCOUNT = 0.99
TAU = 0.005


class TetrisTrainingService(TrainingService[DeepQNetwork, TetrisOfflineMetrics, TetrisModelService, TetrisDataService]):
    modelInitArgs: dict[str, Any]
    optimizerInitialArgs: dict[str, Any]

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
        self.optimizerInitialArgs = {"lr": 1e-4, "amsgrad": True}
        self.policyModel = None
        self.optimizer = None

    def modelFactory(self) -> DeepQNetwork:
        return DeepQNetwork(**self.modelInitArgs)

    def optimizerFactory(self) -> Optimizer:
        if self.policyModel is None:
            raise RuntimeError("self.policyModel is not initialized")
        return AdamW(self.policyModel.parameters(), **self.optimizerInitialArgs)

    def coldStart(self, modelTag: str) -> int:
        self.policyModel = self.modelFactory()
        self.targetModel = self.modelFactory()
        self.optimizer = self.optimizerFactory()
        return self.modelService.publishNewModel(
            modelTag=modelTag,
            policyModel=self.policyModel,
            targetModel=self.targetModel,
            optimizer=self.optimizer,
        )

    def retrainAndPublish(self, modelTag: str, version: int, batchSize: int, numBatches: int) -> TetrisOfflineMetrics:
        if numBatches <= 0:
            raise ValueError("numBatches must be positive")
        entry = self.modelService.getModelEntry(modelTag, version)
        self.policyModel = self.modelFactory()
        self.policyModel.load_state_dict(torch.load(entry.dbKey.policyModelLocation))
        self.targetModel = self.modelFactory()
        self.targetModel.load_state_dict(torch.load(entry.dbKey.targetModelLocation))
        self.optimizer = self.optimizerFactory()
        self.optimizer.load_state_dict(torch.load(entry.dbKey.optimizerLocation))

        trainingLosses: list[float] = []
        for _ in range(numBatches):
            trainLoss = self._performBackpropOnBatch(batchSize)
            trainingLosses.append(trainLoss)
            self._softUpdateTargetModel()

        self.modelService.pushBatchLosses(entry.dbKey, trainingLosses)
        self.modelService.updateModel(
            entry.dbKey,
            policyModel=self.policyModel,
            targetModel=self.targetModel,
            optimizer=self.optimizer,
            numBatchesTrained=numBatches,
        )

        return TetrisOfflineMetrics.fromList(trainingLosses)

    def _performBackpropOnBatch(self, batchSize: int) -> float:
        if batchSize <= 0:
            raise ValueError("batchSize must be positive")
        if self.policyModel is None or self.targetModel is None:
            raise RuntimeError("Policy model or target model not initialized")
        batch = self.dataService.sample(batchSize)
        loss = self._getBatchLoss(batch)

        if self.optimizer is None:
            raise RuntimeError("Optimizer not initialized")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad.clip_grad_value_(self.policyModel.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def _getBatchLoss(self, batch: Sequence[Transition[TetrisState, TetrisAction]]) -> Tensor:
        if self.policyModel is None or self.targetModel is None:
            raise RuntimeError("Policy model or target model not initialized")
        nonFinalMask = torch.tensor(
            tuple(map(lambda s: not s.state.isTerminal, batch)), device=self.device, dtype=torch.bool
        )
        nonFinalNextStates = torch.cat([elt.newState.toDqnInput() for elt in batch if not elt.state.isTerminal])

        stateBatch = torch.cat([elt.state.toDqnInput() for elt in batch])
        actionBatch = torch.tensor([TetrisAgent.possibleActions.index(elt.action) for elt in batch])
        rewardBatch = torch.tensor([elt.reward for elt in batch])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        stateActionValues = self.policyModel(stateBatch).gather(1, actionBatch.reshape(-1, 1)).reshape(-1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        nextStateValues = torch.zeros(len(batch), device=self.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = self.targetModel(nonFinalNextStates).max(1)[0]

        # Compute expected Q values
        expectedStateActionValues = (nextStateValues * FUTURE_REWARDS_DISCOUNT) + rewardBatch

        # Compute Training loss
        criterion = L1Loss()

        return criterion(stateActionValues, expectedStateActionValues)

    def _softUpdateTargetModel(self) -> None:
        if self.policyModel is None or self.targetModel is None:
            raise RuntimeError("Policy model or target model not initialized")
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        targetModelStateDict = self.targetModel.state_dict()
        policyModelStateDict = self.policyModel.state_dict()
        for key in policyModelStateDict:
            targetModelStateDict[key] = policyModelStateDict[key] * TAU + targetModelStateDict[key] * (1 - TAU)
        self.targetModel.load_state_dict(targetModelStateDict)
