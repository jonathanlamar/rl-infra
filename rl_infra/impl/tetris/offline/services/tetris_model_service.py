from __future__ import annotations

import os

import torch

from rl_infra.impl.tetris.offline.models.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.services.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.config import MODEL_ENTRY_PATH, MODEL_ROOT_PATH, MODEL_WEIGHTS_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisOnlineMetrics
from rl_infra.types.offline import ModelDbEntry, ModelDbKey, ModelService, ModelType, SqliteConnection

TetrisModelDbRow = tuple[str, str, str, int, int, float | None, float | None]


class TetrisModelDbEntry(ModelDbEntry[TetrisOnlineMetrics]):
    modelDbKey: ModelDbKey
    modelLocation: str
    numEpochsPlayed: int = 0
    numBatchesTrained: int = 0
    onlinePerformance: TetrisOnlineMetrics = TetrisOnlineMetrics()

    @staticmethod
    def fromDbRow(row: TetrisModelDbRow) -> TetrisModelDbEntry:
        # TODO: This might be doable from pydantic builtins
        key = ModelDbKey(modelType=ModelType[row[0]], modelTag=row[1])
        epochMetrics = TetrisOnlineMetrics(avgEpochLength=row[5], avgEpochScore=row[6])
        return TetrisModelDbEntry(
            modelDbKey=key,
            modelLocation=row[2],
            numEpochsPlayed=row[3],
            numBatchesTrained=row[4],
            onlinePerformance=epochMetrics,
        )


class TetrisModelService(ModelService[DeepQNetwork, TetrisModelDbEntry, TetrisOnlineMetrics]):
    def __init__(self) -> None:
        self.dbPath = f"{DB_ROOT_PATH}/model.db"
        self.modelWeightsPathStub = f"{DB_ROOT_PATH}/models"
        self.deployModelRootPath = MODEL_ROOT_PATH
        self.deployModelWeightsPath = MODEL_WEIGHTS_PATH
        self.deployModelEntryPath = MODEL_ENTRY_PATH
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS models (
                    model_type TEXT NOT NULL,
                    model_tag TEXT NOT NULL,
                    model_location TEXT NOT NULL,
                    num_epochs_played INTEGER NOT NULL,
                    num_batches_trained INTEGER NOT NULL,
                    avg_epoch_length REAL,
                    avg_epoch_score REAL,
                    PRIMARY KEY(model_type, model_tag)
                );"""
            )

    def getModelEntry(self, modelTag: str, modelType: ModelType) -> TetrisModelDbEntry:
        key = ModelDbKey(modelTag=modelTag, modelType=modelType)
        entry = self._fetchEntry(key)
        if entry is None:
            raise KeyError(f"Could not find model with tag {modelTag} and type {modelType}")
        return entry

    def deployModel(self) -> None:
        entry = self._getBestModel(ModelType.ACTOR)
        if not os.path.exists(self.deployModelRootPath):
            os.makedirs(self.deployModelRootPath)
        os.system(f"cp {entry.modelLocation} {self.deployModelWeightsPath}")
        with open(self.deployModelEntryPath, "w") as f:
            f.write(entry.json())

    def updateModel(
        self,
        key: ModelDbKey,
        model: DeepQNetwork | None = None,
        numEpochsPlayed: int | None = None,
        numBatchesTrained: int | None = None,
        onlinePerformance: TetrisOnlineMetrics | None = None,
    ) -> None:
        entry = TetrisModelDbEntry(
            modelDbKey=key,
            modelLocation=self._generateWeightsLocation(key),
            numEpochsPlayed=numEpochsPlayed or 0,
            numBatchesTrained=numBatchesTrained or 0,
            onlinePerformance=onlinePerformance or TetrisOnlineMetrics(),
        )
        maybeExistingEntry = self._fetchEntry(key)
        if maybeExistingEntry is not None:
            entry = maybeExistingEntry.updateWithNewValues(entry)
        if model is not None:
            self._writeWeights(key, model)
        self._upsertToTable(entry)

    def _getBestModel(self, modelType: ModelType) -> TetrisModelDbEntry:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute(
                f"SELECT * FROM models WHERE model_type = '{modelType}' ORDER BY avg_epoch_length DESC;"
            ).fetchone()
        if res is None:
            raise KeyError(f"No best model found with model type {modelType}")
        return TetrisModelDbEntry.fromDbRow(res)

    def _fetchEntry(self, modelDbKey: ModelDbKey) -> TetrisModelDbEntry | None:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute(
                f"""SELECT * FROM models
                WHERE model_type = '{modelDbKey.modelType}' AND model_tag = '{modelDbKey.modelTag}';"""
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
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""INSERT INTO models (
                    model_type,
                    model_tag,
                    model_location,
                    num_epochs_played,
                    num_batches_trained,
                    avg_epoch_length,
                    avg_epoch_score
                ) VALUES (
                    '{entry.modelDbKey.modelType}',
                    '{entry.modelDbKey.modelTag}',
                    '{entry.modelLocation}',
                    {entry.numEpochsPlayed},
                    {entry.numBatchesTrained},
                    {epochLength},
                    {epochScore}
                )
                ON CONFLICT (model_type, model_tag)
                DO UPDATE SET
                    model_location=excluded.model_location,
                    num_epochs_played=excluded.num_epochs_played,
                    num_batches_trained=excluded.num_batches_trained,
                    avg_epoch_length=excluded.avg_epoch_length,
                    avg_epoch_score=excluded.avg_epoch_score;"""
            )

    def _writeWeights(self, key: ModelDbKey, model: DeepQNetwork) -> None:
        location = self._generateWeightsLocation(key)
        if not os.path.exists(os.path.dirname(location)):
            os.makedirs(os.path.dirname(location))
        torch.save(model.state_dict(), location)

    def _generateWeightsLocation(self, modelDbKey: ModelDbKey) -> str:
        directory = f"{self.modelWeightsPathStub}/{modelDbKey.modelType}/{modelDbKey.modelTag}"
        return f"{directory}/weights.pt"
