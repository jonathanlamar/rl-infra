from typing import Any

import torch
from tetris.config import BOARD_SIZE

from rl_infra.impl.tetris.offline.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.tetris_data_service import TetrisDataService
from rl_infra.impl.tetris.offline.tetris_model_service import TetrisModelService, TetrisOfflineMetrics
from rl_infra.impl.tetris.online.tetris_agent import TetrisAgent
from rl_infra.types.offline.training_service import TrainingService


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

    def coldStart(self, modelTag: str):
        self.modelService.publishNewModel(
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

        losses = []
        for _ in range(numBatches):
            actor, loss = self._performBackpropOnBatch(actor, critic, batchSize)
            losses.append(loss)
            critic = self._softUpdateCritic(actor, critic)

        enumLosses = list([(num + entry.numBatchesTrained, loss) for num, loss in enumerate(losses)])
        self.modelService.pushBatchLosses(entry.dbKey, enumLosses)
        self.modelService.updateModel(entry.dbKey, actorModel=actor, criticModel=critic, numBatchesTrained=numBatches)

        return TetrisOfflineMetrics.fromList(losses)

    def _performBackpropOnBatch(
        self, actor: DeepQNetwork, critic: DeepQNetwork, batchSize: int
    ) -> tuple[DeepQNetwork, float]:
        if batchSize <= 0:
            raise ValueError("batchSize must be positive")
        batch = self.dataService.sample(batchSize)
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
        nextStateValues = torch.zeros(batchSize, device=self.device)
        with torch.no_grad():
            nextStateValues[nonFinalMask] = critic(nonFinalNextStates).max(1)[0]

        # Compute expected Q values
        # TODO: Not sure where to stash these constants yet.  Just putting them here for now.
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
        torch.nn.utils.clip_grad.clip_grad_value_(actor.parameters(), 100)
        optimizer.step()

        return actor, loss.item()

    def _softUpdateCritic(self, actor, critic) -> DeepQNetwork:
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        TAU = 0.005
        criticStateDict = critic.state_dict()
        actorStateDict = actor.state_dict()
        for key in actorStateDict:
            criticStateDict[key] = actorStateDict[key] * TAU + criticStateDict[key] * (1 - TAU)
        critic.load_state_dict(criticStateDict)

        return critic