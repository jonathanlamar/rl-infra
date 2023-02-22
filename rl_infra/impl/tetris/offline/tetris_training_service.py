from typing import Any, Sequence

import torch
from tetris.config import BOARD_SIZE
from torch import Tensor
from torch.nn import L1Loss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rl_infra.impl.tetris.offline.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService, TetrisOfflineMetrics
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState
from rl_infra.types.offline.training_service import TrainingService
from rl_infra.types.online.transition import Transition

GAMMA = 0.1
TAU = 0.005


class TetrisTrainingService(TrainingService[DeepQNetwork, TetrisOfflineMetrics]):
    modelService: TetrisModelService
    dataService: TetrisDataService
    device: torch.device
    modelInitArgs: dict[str, Any]
    optimizer: Optimizer | None
    scheduler: ReduceLROnPlateau | None
    actor: DeepQNetwork | None
    critic: DeepQNetwork | None

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
        self.optimizerInitialArgs = {
            "lr": 0.1,
            "momentum": 0.9,
            "dampening": 0.1,
            "weight_decay": 0,
        }
        self.schedulerInitialArgs = {"mode": "min", "patience": 5, "verbose": True}

    def modelFactory(self) -> DeepQNetwork:
        return DeepQNetwork(**self.modelInitArgs)

    def optimizerFactory(self) -> Optimizer:
        if self.actor is None:
            raise RuntimeError("self.actor is not initialized")
        return SGD(self.actor.parameters(), **self.optimizerInitialArgs)

    def schedulerFactory(self) -> ReduceLROnPlateau:
        if self.optimizer is None:
            raise RuntimeError("self.optimizer is not initialized")
        return ReduceLROnPlateau(optimizer=self.optimizer, **self.schedulerInitialArgs)

    def coldStart(self, modelTag: str) -> int:
        self.actor = self.modelFactory()
        self.critic = self.modelFactory()
        self.optimizer = self.optimizerFactory()
        self.scheduler = self.schedulerFactory()
        return self.modelService.publishNewModel(
            modelTag=modelTag,
            actorModel=self.actor,
            criticModel=self.critic,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def retrainAndPublish(self, modelTag: str, version: int, batchSize: int, numBatches: int) -> TetrisOfflineMetrics:
        if numBatches <= 0:
            raise ValueError("numBatches must be positive")
        entry = self.modelService.getModelEntry(modelTag, version)
        self.actor = self.modelFactory()
        self.actor.load_state_dict(torch.load(entry.dbKey.actorLocation))
        self.critic = self.modelFactory()
        self.critic.load_state_dict(torch.load(entry.dbKey.criticLocation))

        self.optimizer = self.optimizerFactory()
        self.optimizer.load_state_dict(torch.load(entry.dbKey.optimizerLocation))
        self.scheduler = self.schedulerFactory()
        self.scheduler.load_state_dict(torch.load(entry.dbKey.schedulerLocation))

        trainingLosses: list[float] = []
        for _ in range(numBatches):
            trainLoss = self._performBackpropOnBatch(batchSize)
            trainingLosses.append(trainLoss)
            self._softUpdateCritic()

        avgTrainingLoss = sum(trainingLosses) / numBatches
        self.scheduler.step(avgTrainingLoss)
        self.modelService.pushBatchLosses(entry.dbKey, trainingLosses)
        self.modelService.updateModel(
            entry.dbKey,
            actorModel=self.actor,
            criticModel=self.critic,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            numBatchesTrained=numBatches,
        )

        return TetrisOfflineMetrics.fromList(trainingLosses)

    def _performBackpropOnBatch(self, batchSize: int) -> float:
        if batchSize <= 0:
            raise ValueError("batchSize must be positive")
        if self.actor is None or self.critic is None:
            raise RuntimeError("Actor or critic not initialized")
        batch = self.dataService.sample(batchSize)
        loss = self._getBatchLoss(batch)

        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer or scheduler not initialized")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad.clip_grad_value_(self.actor.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def _getBatchLoss(self, batch: Sequence[Transition[TetrisState, TetrisAction]]) -> Tensor:
        if self.actor is None or self.critic is None:
            raise RuntimeError("Actor or critic not initialized")
        nonFinalMask = torch.tensor(tuple(map(lambda s: not s.isTerminal, batch)), device=self.device, dtype=torch.bool)
        nonFinalNextStates = torch.cat([elt.newState.toDqnInput() for elt in batch if not elt.isTerminal])

        stateBatch = torch.cat([elt.state.toDqnInput() for elt in batch])
        actionBatch = torch.tensor([TetrisAgent.possibleActions.index(elt.action) for elt in batch])
        rewardBatch = torch.tensor([elt.reward for elt in batch])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        stateActionValues = self.actor(stateBatch).gather(1, actionBatch.reshape(-1, 1)).reshape(-1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        nextStateValues = torch.zeros(len(batch), device=self.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = self.critic(nonFinalNextStates).max(1)[0]

        # Compute expected Q values
        expectedStateActionValues = (nextStateValues * GAMMA) + rewardBatch

        # Compute Training loss
        criterion = L1Loss()
        return criterion(stateActionValues, expectedStateActionValues)

    def _softUpdateCritic(self) -> None:
        if self.actor is None or self.critic is None:
            raise RuntimeError("Actor or critic not initialized")
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        criticStateDict = self.critic.state_dict()
        actorStateDict = self.actor.state_dict()
        for key in actorStateDict:
            criticStateDict[key] = actorStateDict[key] * TAU + criticStateDict[key] * (1 - TAU)
        self.critic.load_state_dict(criticStateDict)
