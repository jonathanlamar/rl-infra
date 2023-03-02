from typing import Protocol, TypeVar

import torch
from torch.optim import Optimizer

from rl_infra.types.base_types import Metrics
from rl_infra.types.offline.data_service import DataService
from rl_infra.types.offline.model_service import ModelService

Model = TypeVar("Model", bound=torch.nn.Module, covariant=True)
OfflineMetrics = TypeVar("OfflineMetrics", bound=Metrics, covariant=True)
MService = TypeVar("MService", bound=ModelService)
DService = TypeVar("DService", bound=DataService)


class TrainingService(Protocol[Model, OfflineMetrics, MService, DService]):
    modelService: MService
    dataService: DService
    device: torch.device
    optimizer: Optimizer | None
    policyModel: Model | None
    targetModel: Model | None

    def modelFactory(self) -> Model:
        ...

    def optimizerFactory(self) -> Model:
        ...

    def coldStart(self, modelTag: str) -> int:
        ...

    def retrainAndPublish(self, modelTag: str, version: int, batchSize: int, numBatches: int) -> OfflineMetrics:
        ...
