import os
from abc import ABC
from enum import Enum
from typing import Generic, Protocol, TypeVar

import torch

from rl_infra.types.base_types import SerializableDataClass

Model = TypeVar("Model", bound=torch.nn.Module, contravariant=True)
Metrics = TypeVar("Metrics", bound=SerializableDataClass, contravariant=True)


class ModelType(str, Enum):
    ACTOR = "ACTOR"
    CRITIC = "CRITIC"


class ModelDbKey(SerializableDataClass):
    modelType: ModelType
    modelTag: str


class ModelDbEntry(ABC, SerializableDataClass, Generic[Metrics]):
    modelDbKey: ModelDbKey
    modelLocation: str
    onlinePerformance: Metrics


# TODO: Implement performance monitoring, versioning?
class ModelService(Protocol[Model, Metrics]):
    dbPath: str
    modelWeightsPathStub: str

    def publishModel(self, model: Model, key: ModelDbKey) -> None:
        ...

    def updateModelMetrics(self, modelDbKey: ModelDbKey, metrics: Metrics) -> None:
        ...

    def deployModel(self) -> None:
        ...

    def _generateWeightsLocation(self, modelDbKey: ModelDbKey) -> str:
        directory = f"{self.modelWeightsPathStub}/{modelDbKey.modelType}/{modelDbKey.modelTag}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        return f"{directory}/weights.pt"
