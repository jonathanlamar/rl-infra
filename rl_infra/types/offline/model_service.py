from __future__ import annotations

from abc import ABC
from typing import Generic, Protocol, TypeVar

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Self

from rl_infra.types.base_types import Metrics, SerializableDataClass


class ModelDbKey(SerializableDataClass):
    tag: str
    version: int
    weightsLocation: str

    @property
    def actorLocation(self) -> str:
        return f"{self.weightsLocation}/actor.pt"

    @property
    def criticLocation(self) -> str:
        return f"{self.weightsLocation}/critic.pt"

    @property
    def optimizerLocation(self) -> str:
        return f"{self.weightsLocation}/optimizer.pt"

    @property
    def schedulerLocation(self) -> str:
        return f"{self.weightsLocation}/scheduler.pt"


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
        actorModel: Model | None,
        criticModel: Model | None,
        optimizer: Optimizer | None,
        scheduler: ReduceLROnPlateau | None,
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
        actorModel: Model | None,
        criticModel: Model | None,
        optimizer: Optimizer | None,
        scheduler: ReduceLROnPlateau | None,
        numEpochsPlayed: int | None,
        numBatchesTrained: int | None,
        onlinePerformance: OnlineMetrics | None,
        offlinePerformance: OfflineMetrics | None,
    ) -> None:
        ...

    def pushBatchLosses(self, modelKey: ModelDbKey, losses: list[float]) -> None:
        ...
