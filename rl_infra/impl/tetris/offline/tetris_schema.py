from __future__ import annotations

from typing import Any, NamedTuple

from pydantic.utils import GetterDict

from rl_infra.types.offline.schema import (
    ModelDbEntry,
    ModelDbKey,
    OfflineMetrics,
    OfflineMetricsDbEntry,
    OnlineMetrics,
    OnlineMetricsDbEntry,
)


class TetrisOnlineMetrics(OnlineMetrics):
    numMoves: int
    score: int


class TetrisOfflineMetrics(OfflineMetrics):
    numBatchesTrained: int
    avgBatchLoss: float
    validationEpisodeId: int
    valEpisodeAvgMaxQ: float  # TODO: This needs a better name


class TetrisModelDbRow(NamedTuple):
    tag: str
    version: int
    weightsLocation: str
    numEpisodesPlayed: int
    numEpochsTrained: int
    avgEpisodeLength: float | None
    avgEpisodeScore: float | None
    recencyWeightedAvgLoss: float | None
    recencyWeightedAvgValidationQ: float | None


class TetrisOfflineMetricsDbRow(NamedTuple):
    tag: str
    version: int
    epochNumber: int
    numBatchesTrained: int
    avgBatchLoss: float
    validationEpisodeId: str
    valEpisodeAvgMaxQ: float


class TetrisOnlineMetricsDbRow(NamedTuple):
    tag: str
    version: int
    episodeNumber: int
    numMoves: int
    score: int


class TetrisModelDbGetterDict(GetterDict):
    """Special logic for parsing Tetris model database rows"""

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self._obj, TetrisModelDbRow):
            if key == "modelDbKey":
                return ModelDbKey.from_orm(self._obj)
            if key == "onlinePerformance":
                return TetrisOnlineMetrics.from_orm(self._obj)
        if isinstance(self._obj, TetrisOfflineMetricsDbRow):
            if key == "modelDbKey":
                return ModelDbKey.from_orm(self._obj)
            if key == "offlineMetrics":
                return TetrisOfflineMetrics.from_orm(self._obj)
        if isinstance(self._obj, TetrisOnlineMetricsDbRow):
            if key == "modelDbKey":
                return ModelDbKey.from_orm(self._obj)
            if key == "onlineMetrics":
                return TetrisOfflineMetrics.from_orm(self._obj)
        return super().get(key, default)


class TetrisOnlineMetricsDbEntry(OnlineMetricsDbEntry[TetrisOnlineMetrics]):
    onlineMetrics: TetrisOnlineMetrics

    class Config(ModelDbEntry.Config):
        """pydantic config class"""

        getter_dict = TetrisModelDbGetterDict


class TetrisOfflineMetricsDbEntry(OfflineMetricsDbEntry[TetrisOfflineMetrics]):
    offlineMetrics: TetrisOfflineMetrics

    class Config(ModelDbEntry.Config):
        """pydantic config class"""

        getter_dict = TetrisModelDbGetterDict


class TetrisModelDbEntry(ModelDbEntry):
    avgEpisodeLength: float | None = None
    avgEpisodeScore: float | None = None
    recencyWeightedAvgLoss: float | None = None
    recencyWeightedAvgValidationQ: float | None = None

    def updateWithNewValues(self, other: TetrisModelDbEntry) -> TetrisModelDbEntry:
        if self.modelDbKey != other.modelDbKey:
            raise KeyError("Cannot average incompatible model tags")

        # These should never be in danger of invalid values, but it doesn't hurt.
        assert self.numEpisodesPlayed >= 0
        assert self.numEpochsTrained >= 0
        assert other.numEpisodesPlayed >= 0
        assert other.numEpochsTrained >= 0
        assert (self.numEpisodesPlayed == 0) == (self.avgEpisodeLength is None)
        assert (self.numEpisodesPlayed == 0) == (self.avgEpisodeScore is None)
        assert (self.numEpochsTrained == 0) == (self.recencyWeightedAvgLoss is None)
        assert (self.numEpochsTrained == 0) == (self.recencyWeightedAvgValidationQ is None)

        # This function should never fail if the above assert pass
        def weightedAvg(avg1: float | None, count1: int, avg2: float | None, count2: int) -> float | None:
            if avg1 is None:
                return avg2
            if avg2 is None:
                return avg1
            return (avg1 * count1 + avg2 * count2) / (count1 + count2)

        return self.__class__(
            modelDbKey=self.modelDbKey,
            numEpisodesPlayed=self.numEpisodesPlayed + other.numEpisodesPlayed,
            numEpochsTrained=self.numEpochsTrained + other.numEpochsTrained,
            avgEpisodeLength=weightedAvg(
                self.avgEpisodeLength, self.numEpisodesPlayed, other.avgEpisodeLength, other.numEpisodesPlayed
            ),
            avgEpisodeScore=weightedAvg(
                self.avgEpisodeScore, self.numEpisodesPlayed, other.avgEpisodeScore, other.numEpisodesPlayed
            ),
            recencyWeightedAvgLoss=weightedAvg(
                self.recencyWeightedAvgLoss, self.numEpochsTrained, other.recencyWeightedAvgLoss, other.numEpochsTrained
            ),
            recencyWeightedAvgValidationQ=weightedAvg(
                self.recencyWeightedAvgValidationQ,
                self.numEpochsTrained,
                other.recencyWeightedAvgValidationQ,
                other.numEpochsTrained,
            ),
        )

    @staticmethod
    def fromMetrics(
        modelDbKey: ModelDbKey,
        onlineMetrics: TetrisOnlineMetrics | None = None,
        offlineMetrics: TetrisOfflineMetrics | None = None,
    ) -> TetrisModelDbEntry:
        return TetrisModelDbEntry(
            modelDbKey=modelDbKey,
            numEpisodesPlayed=int(onlineMetrics is not None),
            numEpochsTrained=int(offlineMetrics is not None),
            avgEpisodeLength=None if onlineMetrics is None else onlineMetrics.numMoves,
            avgEpisodeScore=None if onlineMetrics is None else onlineMetrics.score,
            recencyWeightedAvgLoss=None if offlineMetrics is None else offlineMetrics.avgBatchLoss,
            recencyWeightedAvgValidationQ=None if offlineMetrics is None else offlineMetrics.valEpisodeAvgMaxQ,
        )

    class Config(ModelDbEntry.Config):
        """pydantic config class"""

        getter_dict = TetrisModelDbGetterDict
