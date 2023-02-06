from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Generic, Protocol, TypeVar

import torch
from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.online.environment import ModelOnlineMetrics


class ModelType(str, Enum):
    ACTOR = "ACTOR"
    CRITIC = "CRITIC"


class ModelDbKey(SerializableDataClass):
    modelType: ModelType
    modelTag: str


Metrics = TypeVar("Metrics", bound=ModelOnlineMetrics, contravariant=True)


class ModelDbEntry(ABC, SerializableDataClass, Generic[Metrics]):
    modelDbKey: ModelDbKey
    modelLocation: str
    numEpochsPlayed: int
    numBatchesTrained: int
    onlinePerformance: Metrics

    def updateWithNewValues(self, other: Self) -> Self:
        if self.modelDbKey != other.modelDbKey or self.modelLocation != other.modelLocation:
            raise KeyError("Cannot average incompatible model tags")

        return self.__class__(
            modelDbKey=self.modelDbKey,
            modelLocation=self.modelLocation,
            numEpochsPlayed=self.numEpochsPlayed + other.numEpochsPlayed,
            numBatchesTrained=self.numBatchesTrained + other.numBatchesTrained,
            onlinePerformance=self.onlinePerformance.updateWithNewValues(other.onlinePerformance),
        )


Model = TypeVar("Model", bound=torch.nn.Module, contravariant=True)
Entry = TypeVar("Entry", bound=ModelDbEntry, covariant=True)


# TODO: Implement performance monitoring, versioning?
class ModelService(Protocol[Model, Entry, Metrics]):
    def publishNewModel(self, model: Model, key: ModelDbKey) -> None:
        self.updateModel(key, model)

    def getModelEntry(self, modelTag: str, modelType: ModelType) -> Entry:
        ...

    def deployModel(self) -> None:
        ...

    def updateModel(
        self,
        key: ModelDbKey,
        model: Model | None = None,
        numEpochsPlayed: int | None = None,
        numBatchesTrained: int | None = None,
        onlinePerformance: Metrics | None = None,
    ) -> None:
        ...
