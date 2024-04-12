from typing import Protocol, TypeVar

import torch
from torch.optim import Optimizer

from rl_infra.types.offline.data_service import DataService
from rl_infra.types.offline.model_service import ModelDbKey, ModelService

Model = TypeVar("Model", bound=torch.nn.Module, covariant=True)
MService = TypeVar("MService", bound=ModelService)
DService = TypeVar("DService", bound=DataService)


class TrainingService(Protocol[Model, MService, DService]):
    modelService: MService
    dataService: DService
    device: torch.device
    optimizer: Optimizer | None
    policyModel: Model | None
    targetModel: Model | None

    def modelFactory(self) -> Model: ...

    def optimizerFactory(self) -> Model: ...

    def coldStart(self, modelTag: str) -> int: ...

    def retrainAndPublish(
        self,
        modelDbKey: ModelDbKey,
        episodeNumber: int,
        batchSize: int,
        numBatches: int,
        validationEpisodeId: int | None = None,
    ) -> None: ...

    def validateOnEpisode(self, validationEpisodeId: int | None = None) -> tuple[float, int]: ...
