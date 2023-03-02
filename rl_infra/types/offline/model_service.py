from __future__ import annotations

from abc import ABC
from typing import Generic, Protocol, TypeVar

from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import Self

from rl_infra.types.base_types import Metrics, SerializableDataClass


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


OnlineMetrics = TypeVar("OnlineMetrics", bound=Metrics, contravariant=True)
OfflineMetrics = TypeVar("OfflineMetrics", bound=Metrics, contravariant=True)


class ModelDbEntry(ABC, SerializableDataClass, Generic[OnlineMetrics, OfflineMetrics]):
    dbKey: ModelDbKey
    numEpochsPlayed: int
    numBatchesTrained: int
    onlinePerformance: OnlineMetrics
    offlinePerformance: OfflineMetrics

    def updateWithNewValues(self, other: Self) -> Self:
        if self.dbKey != other.dbKey:
            raise KeyError("Cannot average incompatible model tags")

        return self.__class__(
            dbKey=self.dbKey,
            numEpochsPlayed=self.numEpochsPlayed + other.numEpochsPlayed,
            numBatchesTrained=self.numBatchesTrained + other.numBatchesTrained,
            onlinePerformance=self.onlinePerformance.updateWithNewValues(other.onlinePerformance),
            offlinePerformance=self.offlinePerformance.updateWithNewValues(other.offlinePerformance),
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
        ...

    def getLatestVersionKey(self, modelTag: str) -> ModelDbKey | None:
        ...

    def getModelEntry(self, modelTag: str, version: int) -> ModelDbEntry[OnlineMetrics, OfflineMetrics]:
        ...

    def deployModel(self, modelTag: str | None, version: int | None) -> int:
        ...

    def updateModel(
        self,
        key: ModelDbKey,
        policyModel: Model | None,
        targetModel: Model | None,
        optimizer: Optimizer | None,
        numEpochsPlayed: int | None,
        numBatchesTrained: int | None,
        onlinePerformance: OnlineMetrics | None,
        offlinePerformance: OfflineMetrics | None,
    ) -> None:
        ...

    def pushBatchLosses(self, modelKey: ModelDbKey, losses: list[float]) -> None:
        ...
