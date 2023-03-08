from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.online.agent import OfflineMetrics, OnlineMetrics


class ModelDbKey(SerializableDataClass):
    tag: str
    version: int
    weightsLocation: str

    @property
    def policyModelLocation(self) -> str:
        return f"{self.weightsLocation}/policyModel.pt"

    @property
    def targetModelLocation(self) -> str:
        return f"{self.weightsLocation}/targetModel.pt"

    @property
    def optimizerLocation(self) -> str:
        return f"{self.weightsLocation}/optimizer.pt"


OnM_co = TypeVar("OnM_co", bound=OnlineMetrics, covariant=True)
OffM_co = TypeVar("OffM_co", bound=OfflineMetrics, covariant=True)


class OfflineMetricsDbEntry(SerializableDataClass, Generic[OffM_co]):
    modelDbKey: ModelDbKey
    offlineMetrics: OffM_co


class OnlineMetricsDbEntry(SerializableDataClass, Generic[OnM_co]):
    modelDbKey: ModelDbKey
    onlineMetrics: OnM_co


class ModelDbEntry(SerializableDataClass):
    modelDbKey: ModelDbKey
    numEpisodesPlayed: int
    numEpochsTrained: int

    def updateWithNewValues(self, other: Self) -> Self:
        if self.modelDbKey != other.modelDbKey:
            raise KeyError("Cannot average incompatible model tags")

        return self.__class__(
            modelDbKey=self.modelDbKey,
            numEpisodesPlayed=self.numEpisodesPlayed + other.numEpisodesPlayed,
            numEpochsTrained=self.numEpochsTrained + other.numEpochsTrained,
        )


Model = TypeVar("Model", bound=Module, contravariant=True)


class ModelService(Protocol[Model, OnlineMetrics, OfflineMetrics]):
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

    def getModelKey(self, modelTag: str, version: int) -> ModelDbKey:
        ...

    def getLatestVersionKey(self, modelTag: str) -> ModelDbKey | None:
        ...

    def getModelEntry(self, key: ModelDbKey) -> ModelDbEntry | None:
        ...

    def deployModel(self, key: ModelDbKey) -> None:
        ...

    def updateModel(
        self,
        key: ModelDbKey,
        policyModel: Model | None,
        targetModel: Model | None,
        optimizer: Optimizer | None,
    ) -> None:
        ...

    def publishOnlineMetrics(self, key: ModelDbKey, onlineMetrics: OnlineMetrics) -> None:
        ...

    def publishOfflineMetrics(self, key: ModelDbKey, offlineMetrics: OfflineMetrics) -> None:
        ...
