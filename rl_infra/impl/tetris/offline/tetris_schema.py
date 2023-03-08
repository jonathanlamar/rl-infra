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
    numBatchesTrained: int
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
            if key == "dbKey":
                return ModelDbKey.from_orm(self._obj)
            if key == "onlinePerformance":
                return TetrisOnlineMetrics.from_orm(self._obj)
        if isinstance(self._obj, TetrisOfflineMetricsDbRow):
            if key == "dbKey":
                return ModelDbKey.from_orm(self._obj)
            if key == "offlineMetrics":
                return TetrisOfflineMetrics.from_orm(self._obj)
        if isinstance(self._obj, TetrisOnlineMetricsDbRow):
            if key == "dbKey":
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

    @classmethod
    def fromMetrics(
        cls,
        dbKey: ModelDbKey,
        onlineMetrics: TetrisOnlineMetrics | None = None,
        offlineMetrics: TetrisOfflineMetrics | None = None,
    ) -> TetrisModelDbEntry:
        return TetrisModelDbEntry(
            modelDbKey=dbKey,
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
