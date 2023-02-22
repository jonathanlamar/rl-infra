from typing import Protocol, TypeVar

import torch

from rl_infra.types.base_types import Metrics

Model = TypeVar("Model", bound=torch.nn.Module, covariant=True)
OfflineMetrics = TypeVar("OfflineMetrics", bound=Metrics, covariant=True)


class TrainingService(Protocol[Model, OfflineMetrics]):
    def modelFactory(self) -> Model:
        ...

    def coldStart(self, modelTag: str) -> int:
        ...

    def retrainAndPublish(self, modelTag: str, version: int, batchSize: int, numBatches: int) -> OfflineMetrics:
        ...
