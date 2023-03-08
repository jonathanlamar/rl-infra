from abc import ABC
from typing import Generic, TypeVar

from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass


class OnlineMetrics(ABC, SerializableDataClass):
    episodeNumber: int
    # Other metrics vary by implementation


class OfflineMetrics(ABC, SerializableDataClass):
    epochNumber: int
    # Other metrics vary by implementation


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
