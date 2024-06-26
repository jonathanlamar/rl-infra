from __future__ import annotations

from typing import Protocol, TypeVar

from torch.nn import Module
from torch.optim import Optimizer

from rl_infra.types.offline.schema import ModelDbEntry, ModelDbKey, OfflineMetrics, OnlineMetrics

Model = TypeVar("Model", bound=Module, contravariant=True)
OnM = TypeVar("OnM", bound=OnlineMetrics, contravariant=True)
OffM = TypeVar("OffM", bound=OfflineMetrics, contravariant=True)


class ModelService(Protocol[Model, OnM, OffM]):
    def publishNewModel(
        self,
        modelTag: str,
        policyModel: Model | None,
        targetModel: Model | None,
        optimizer: Optimizer | None,
    ) -> int:
        """Publish new version of model with tag modelTag.  Optionally provide weights for policy mode, target model,
        or optimizer.  Otherwise random weights are initialized.  Returns version of newly published model."""
        ...

    def getModelKey(self, modelTag: str, version: int) -> ModelDbKey: ...

    def getLatestVersionKey(self, modelTag: str) -> ModelDbKey | None: ...

    def getModelEntry(self, key: ModelDbKey) -> ModelDbEntry | None: ...

    def deployModel(self, key: ModelDbKey) -> None: ...

    def updateModelWeights(
        self,
        key: ModelDbKey,
        policyModel: Model | None,
        targetModel: Model | None,
        optimizer: Optimizer | None,
    ) -> None: ...

    def publishOnlineMetrics(self, key: ModelDbKey, onlineMetrics: OnM) -> None: ...

    def publishOfflineMetrics(self, key: ModelDbKey, offlineMetrics: OffM) -> None: ...
