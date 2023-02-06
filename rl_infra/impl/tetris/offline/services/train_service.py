from typing import Any

import torch
from tetris.config import BOARD_SIZE

from rl_infra.impl.tetris.offline.models.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.services.data_service import TetrisDataService
from rl_infra.impl.tetris.offline.services.model_service import TetrisModelService
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
        print("Publishing new actor model")
        self.modelService.publishNewModel(
            model=self.modelFactory(),
            key=ModelDbKey(modelType=ModelType.ACTOR, modelTag=modelTag),
        )
        print("Publishing new critic model")
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

        # TODO: Implement training
        # batch = self.dataService.sample(batchSize)

        self.modelService.updateModel(actorEntry.modelDbKey, model=actor, numBatchesTrained=numBatches)
        self.modelService.updateModel(criticEntry.modelDbKey, model=critic, numBatchesTrained=1)
