from typing import Protocol, TypeVar

import torch

from rl_infra.types.base_types import Metrics
from rl_infra.types.offline.data_service import DataService
from rl_infra.types.offline.model_service import ModelService

MService = TypeVar("MService", bound=ModelService, covariant=True)
DService = TypeVar("DService", bound=DataService, covariant=True)
Model = TypeVar("Model", bound=torch.nn.Module, covariant=True)
OfflineMetrics = TypeVar("OfflineMetrics", bound=Metrics, covariant=True)


class TrainingService(Protocol[MService, DService, Model, OfflineMetrics]):
    def modelFactory(self) -> Model:
        ...

    def coldStart(self, modelTag: str) -> None:
        ...

    def retrainAndPublish(self, modelTag: str, version: int, numBatches: int) -> OfflineMetrics:
        ...
