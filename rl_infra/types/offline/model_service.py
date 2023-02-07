from __future__ import annotations

from abc import ABC
from typing import Generic, Protocol, TypeVar

import torch
from typing_extensions import Self

from rl_infra.types.base_types import Metrics, SerializableDataClass


class ModelDbKey(SerializableDataClass):
    tag: str
    version: int


OnlineMetrics = TypeVar("OnlineMetrics", bound=Metrics, contravariant=True)
OfflineMetrics = TypeVar("OfflineMetrics", bound=Metrics, contravariant=True)


class ModelDbEntry(ABC, SerializableDataClass, Generic[OnlineMetrics, OfflineMetrics]):
    dbKey: ModelDbKey
    actorLocation: str
    criticLocation: str
    numEpochsPlayed: int
    numBatchesTrained: int
    onlinePerformance: OnlineMetrics
    offlinePerformance: OfflineMetrics

    def updateWithNewValues(self, other: Self) -> Self:
        if self.dbKey != other.dbKey or self.actorLocation != other.actorLocation:
            raise KeyError("Cannot average incompatible model tags")

        return self.__class__(
            dbKey=self.dbKey,
            actorLocation=self.actorLocation,
            criticLocation=self.criticLocation,
            numEpochsPlayed=self.numEpochsPlayed + other.numEpochsPlayed,
            numBatchesTrained=self.numBatchesTrained + other.numBatchesTrained,
            onlinePerformance=self.onlinePerformance.updateWithNewValues(other.onlinePerformance),
            offlinePerformance=self.offlinePerformance.updateWithNewValues(other.offlinePerformance),
        )


Model = TypeVar("Model", bound=torch.nn.Module, contravariant=True)
Entry = TypeVar("Entry", bound=ModelDbEntry, covariant=True)


class ModelService(Protocol[Model, Entry, OnlineMetrics, OfflineMetrics]):
    def publishNewModel(self, modelTag: str, actorModel: Model | None = None, criticModel: Model | None = None) -> None:
        ...

    def getLatestVersionKey(self, modelTag: str) -> ModelDbKey | None:
        ...

    def getModelEntry(self, modelTag: str, version: int) -> Entry:
        ...

    def deployModel(self) -> None:
        ...

    def updateModel(
        self,
        key: ModelDbKey,
        actorModel: Model | None = None,
        criticModel: Model | None = None,
        numEpochsPlayed: int | None = None,
        numBatchesTrained: int | None = None,
        onlinePerformance: OnlineMetrics | None = None,
        offlinePerformance: OfflineMetrics | None = None,
    ) -> None:
        ...
