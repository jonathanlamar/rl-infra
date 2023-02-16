from typing import Any

import torch
from tetris.config import BOARD_SIZE

from rl_infra.impl.tetris.offline.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService, TetrisOfflineMetrics
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.impl.tetris.online.tetris_transition import TetrisTransition
from rl_infra.types.offline.training_service import TrainingService

GAMMA = 0.1
TAU = 0.005
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0


class TetrisTrainingService(TrainingService):
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

    def coldStart(self, modelTag: str) -> int:
        return self.modelService.publishNewModel(
            modelTag=modelTag,
            actorModel=self.modelFactory(),
            criticModel=self.modelFactory(),
        )

    def retrainAndPublish(self, modelTag: str, version: int, batchSize: int, numBatches: int) -> TetrisOfflineMetrics:
        if numBatches <= 0:
            raise ValueError("numBatches must be positive")
        entry = self.modelService.getModelEntry(modelTag, version)
        actor = self.modelFactory()
        actor.load_state_dict(torch.load(entry.actorLocation))
        critic = self.modelFactory()
        critic.load_state_dict(torch.load(entry.criticLocation))

        trainingLosses = []
        for _ in range(numBatches):
            optimizer = torch.optim.SGD(actor.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            actor, trainLoss = self._performBackpropOnBatch(actor, critic, batchSize, optimizer)
            trainingLosses.append(trainLoss)

            scheduler.step(trainLoss)

            critic = self._softUpdateCritic(actor, critic)

        enumLosses = list([(num + entry.numBatchesTrained, loss) for num, loss in enumerate(trainingLosses)])
        self.modelService.pushBatchLosses(entry.dbKey, enumLosses)
        self.modelService.updateModel(entry.dbKey, actorModel=actor, criticModel=critic, numBatchesTrained=numBatches)

        return TetrisOfflineMetrics.fromList(trainingLosses)

    def _performBackpropOnBatch(
            self, actor: DeepQNetwork, critic: DeepQNetwork, batchSize: int, optimizer: torch.optim.Optimizer
    ) -> tuple[DeepQNetwork, float]:
        if batchSize <= 0:
            raise ValueError("batchSize must be positive")
        batch = self.dataService.sample(batchSize)
        loss = self._getBatchLoss(batch, actor, critic)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad.clip_grad_value_(actor.parameters(), 100)
        optimizer.step()

        return actor, loss.item()

    def _getBatchLoss(self, batch: list[TetrisTransition], actor: DeepQNetwork, critic: DeepQNetwork) -> torch.Tensor:
        nonFinalMask = torch.tensor(tuple(map(lambda s: not s.isTerminal, batch)), device=self.device, dtype=torch.bool)
        nonFinalNextStates = torch.cat([elt.newState.toDqnInput() for elt in batch if not elt.isTerminal])

        stateBatch = torch.cat([elt.state.toDqnInput() for elt in batch])
        actionBatch = torch.tensor([TetrisAgent.possibleActions.index(elt.action) for elt in batch])
        rewardBatch = torch.tensor([elt.reward for elt in batch])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        stateActionValues = actor(stateBatch).gather(1, actionBatch.reshape(-1, 1)).reshape(-1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        nextStateValues = torch.zeros(len(batch), device=self.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = critic(nonFinalNextStates).max(1)[0]

        # Compute expected Q values
        expectedStateActionValues = (nextStateValues * GAMMA) + rewardBatch

        # Compute Huber loss
        criterion = torch.nn.L1Loss()
        return criterion(stateActionValues, expectedStateActionValues)

    def _softUpdateCritic(self, actor: DeepQNetwork, critic: DeepQNetwork) -> DeepQNetwork:
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        criticStateDict = critic.state_dict()
        actorStateDict = actor.state_dict()
        for key in actorStateDict:
            criticStateDict[key] = actorStateDict[key] * TAU + criticStateDict[key] * (1 - TAU)
        critic.load_state_dict(criticStateDict)

        return critic
