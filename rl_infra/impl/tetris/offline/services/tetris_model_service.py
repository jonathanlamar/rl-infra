from __future__ import annotations

import os

import torch

from rl_infra.impl.tetris.offline.models.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.services.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.config import MODEL_ENTRY_PATH, MODEL_ROOT_PATH, MODEL_WEIGHTS_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisOnlineMetrics
from rl_infra.types.base_types import Metrics
from rl_infra.types.offline import ModelDbEntry, ModelDbKey, ModelService, SqliteConnection

TetrisModelDbRow = tuple[str, int, str, str, int, int, float | None, float | None, float | None]


class TetrisOfflineMetrics(Metrics):
    avgHuberLoss: float | None = None

    @classmethod
    def fromList(cls, vals: list[float]) -> TetrisOfflineMetrics:
        if not vals:
            return TetrisOfflineMetrics()
        avgLoss = sum(vals) / len(vals)
        return cls(avgHuberLoss=avgLoss)

    def updateWithNewValues(self, other: TetrisOfflineMetrics) -> TetrisOfflineMetrics:
        return TetrisOfflineMetrics(
            avgHuberLoss=TetrisOfflineMetrics.avgWithoutNone(self.avgHuberLoss, other.avgHuberLoss),
        )


class TetrisModelDbEntry(ModelDbEntry[TetrisOnlineMetrics, TetrisOfflineMetrics]):
    dbKey: ModelDbKey
    actorLocation: str
    criticLocation: str
    numEpochsPlayed: int = 0
    numBatchesTrained: int = 0
    onlinePerformance: TetrisOnlineMetrics = TetrisOnlineMetrics()
    offlinePerformance: TetrisOfflineMetrics = TetrisOfflineMetrics()

    @staticmethod
    def fromDbRow(row: TetrisModelDbRow) -> TetrisModelDbEntry:
        # TODO: This might be doable from pydantic builtins
        key = ModelDbKey(tag=row[0], version=row[1])
        onlineMetrics = TetrisOnlineMetrics(avgEpochLength=row[6], avgEpochScore=row[7])
        offlineMetrics = TetrisOfflineMetrics(avgHuberLoss=row[8])
        return TetrisModelDbEntry(
            dbKey=key,
            actorLocation=row[2],
            criticLocation=row[3],
            numEpochsPlayed=row[4],
            numBatchesTrained=row[5],
            onlinePerformance=onlineMetrics,
            offlinePerformance=offlineMetrics,
        )


class TetrisModelService(ModelService[DeepQNetwork, TetrisModelDbEntry, TetrisOnlineMetrics, TetrisOfflineMetrics]):
    def __init__(self) -> None:
        self.dbPath = f"{DB_ROOT_PATH}/model.db"
        self.modelWeightsPathStub = f"{DB_ROOT_PATH}/models"
        self.deployModelRootPath = MODEL_ROOT_PATH
        self.deployModelWeightsPath = MODEL_WEIGHTS_PATH
        self.deployModelEntryPath = MODEL_ENTRY_PATH
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS models (
                    tag TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    actor_location TEXT NOT NULL,
                    critic_location TEXT NOT NULL,
                    num_epochs_played INTEGER NOT NULL,
                    num_batches_trained INTEGER NOT NULL,
                    avg_epoch_length REAL,
                    avg_epoch_score REAL,
                    avg_huber_loss REAL,
                    PRIMARY KEY(tag, version)
                );"""
            )

    def publishNewModel(
        self, modelTag: str, actorModel: DeepQNetwork | None = None, criticModel: DeepQNetwork | None = None
    ) -> None:
        maybeExistingModelKey = self.getLatestVersionKey(modelTag)
        if maybeExistingModelKey is not None:
            version = maybeExistingModelKey.version
        version = 0
        newModelKey = ModelDbKey(tag=modelTag, version=version)
        self.updateModel(newModelKey, actorModel=actorModel, criticModel=criticModel)

    def getLatestVersionKey(self, modelTag: str) -> ModelDbKey | None:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute(
                f"SELECT tag, version FROM models WHERE tag = '{modelTag}' ORDER BY version DESC"
            ).fetchone()
        if res is None:
            return None
        return ModelDbKey(tag=res[0], version=res[1])

    def getModelEntry(self, modelTag: str, version: int) -> TetrisModelDbEntry:
        key = ModelDbKey(tag=modelTag, version=version)
        entry = self._fetchEntry(key)
        if entry is None:
            raise KeyError(f"Could not find model with tag {modelTag} and version {version}")
        return entry

    def deployModel(self) -> None:
        entry = self._getBestModel()
        if not os.path.exists(self.deployModelRootPath):
            os.makedirs(self.deployModelRootPath)
        os.system(f"cp {entry.actorLocation} {self.deployModelWeightsPath}")
        with open(self.deployModelEntryPath, "w") as f:
            f.write(entry.json())

    def updateModel(
        self,
        key: ModelDbKey,
        actorModel: DeepQNetwork | None = None,
        criticModel: DeepQNetwork | None = None,
        numEpochsPlayed: int | None = None,
        numBatchesTrained: int | None = None,
        onlinePerformance: TetrisOnlineMetrics | None = None,
        offlinePerformance: TetrisOfflineMetrics | None = None,
    ) -> None:
        weightsLocation = self._generateWeightsLocation(key)
        actorWeightsLocation = f"{weightsLocation}/actor.pt"
        criticWeightsLocation = f"{weightsLocation}/critic.pt"
        entry = TetrisModelDbEntry(
            dbKey=key,
            actorLocation=actorWeightsLocation,
            criticLocation=criticWeightsLocation,
            numEpochsPlayed=numEpochsPlayed or 0,
            numBatchesTrained=numBatchesTrained or 0,
            onlinePerformance=onlinePerformance or TetrisOnlineMetrics(),
            offlinePerformance=offlinePerformance or TetrisOfflineMetrics(),
        )
        maybeExistingEntry = self._fetchEntry(key)
        if maybeExistingEntry is not None:
            entry = maybeExistingEntry.updateWithNewValues(entry)
        self._writeWeights(key, actorModel, criticModel)
        self._upsertToTable(entry)

    def _getBestModel(self) -> TetrisModelDbEntry:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute("SELECT * FROM models ORDER BY avg_epoch_length DESC;").fetchone()
        if res is None:
            raise KeyError("No best model found")
        return TetrisModelDbEntry.fromDbRow(res)

    def _fetchEntry(self, modelDbKey: ModelDbKey) -> TetrisModelDbEntry | None:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute(
                f"""SELECT * FROM models WHERE tag = '{modelDbKey.tag}' AND version = {modelDbKey.version};"""
            ).fetchone()
        if res is None:
            return None
        return TetrisModelDbEntry.fromDbRow(res)

    def _upsertToTable(self, entry: TetrisModelDbEntry) -> None:
        epochLength = (
            entry.onlinePerformance.avgEpochLength if entry.onlinePerformance.avgEpochLength is not None else "NULL"
        )
        epochScore = (
            entry.onlinePerformance.avgEpochScore if entry.onlinePerformance.avgEpochScore is not None else "NULL"
        )
        huberLoss = (
            entry.offlinePerformance.avgHuberLoss if entry.offlinePerformance.avgHuberLoss is not None else "NULL"
        )
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""INSERT INTO models (
                    tag,
                    version,
                    actor_location,
                    critic_location,
                    num_epochs_played,
                    num_batches_trained,
                    avg_epoch_length,
                    avg_epoch_score,
                    avg_huber_loss
                ) VALUES (
                    '{entry.dbKey.tag}',
                    '{entry.dbKey.version}',
                    '{entry.actorLocation}',
                    '{entry.criticLocation}',
                    {entry.numEpochsPlayed},
                    {entry.numBatchesTrained},
                    {epochLength},
                    {epochScore},
                    {huberLoss}
                )
                ON CONFLICT (tag, version)
                DO UPDATE SET
                    actor_location=excluded.actor_location,
                    critic_location=excluded.critic_location,
                    num_epochs_played=excluded.num_epochs_played,
                    num_batches_trained=excluded.num_batches_trained,
                    avg_epoch_length=excluded.avg_epoch_length,
                    avg_epoch_score=excluded.avg_epoch_score,
                    avg_huber_loss=excluded.avg_huber_loss;"""
            )

    def _writeWeights(
        self, key: ModelDbKey, actorModel: DeepQNetwork | None = None, criticModel: DeepQNetwork | None = None
    ) -> None:
        location = self._generateWeightsLocation(key)
        if not os.path.exists(location):
            os.makedirs(location)
        if actorModel is not None:
            torch.save(actorModel.state_dict(), f"{location}/actor.pt")
        if criticModel is not None:
            torch.save(criticModel.state_dict(), f"{location}/critic.pt")

    def _generateWeightsLocation(self, modelDbKey: ModelDbKey) -> str:
        return f"{self.modelWeightsPathStub}/{modelDbKey.tag}/{modelDbKey.version}"
