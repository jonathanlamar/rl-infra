from __future__ import annotations

import os
import sqlite3
from enum import Enum

import torch

from rl_infra.base_types import SerializableDataClass
from rl_infra.offline.tetris.models.dqn import DeepQNetwork
from rl_infra.offline.tetris.services.config import DB_ROOT_PATH
from rl_infra.online.impl.tetris.config import MODEL_ROOT_PATH


class ModelType(str, Enum):
    ACTOR = "ACTOR"
    CRITIC = "CRITIC"


ModelDbRow = tuple[str, str, str, int, float, float]


class EpochMetrics(SerializableDataClass):
    numEpochsTrained: int
    avgEpochLength: float
    avgEpochScore: float

    def avgWith(self, other: EpochMetrics) -> EpochMetrics:
        return EpochMetrics(
            numEpochsTrained=self.numEpochsTrained + other.numEpochsTrained,
            avgEpochLength=(self.avgEpochLength + other.avgEpochLength) / 2.0,
            avgEpochScore=(self.avgEpochScore + other.avgEpochScore) / 2.0,
        )


class ModelDbKey(SerializableDataClass):
    modelType: ModelType
    modelTag: str


class ModelDbEntry(SerializableDataClass):
    modelDbKey: ModelDbKey
    modelLocation: str
    onlinePerformance: EpochMetrics

    @staticmethod
    def fromDbRow(row: ModelDbRow) -> ModelDbEntry:
        # TODO: This might be doable from pydantic builtins
        return ModelDbEntry(
            modelDbKey=ModelDbKey(modelType=ModelType[row[0]], modelTag=row[1]),
            modelLocation=row[2],
            onlinePerformance=EpochMetrics(numEpochsTrained=row[3], avgEpochLength=row[4], avgEpochScore=row[5]),
        )


class ModelService:
    rootPath: str
    modelDb: sqlite3.Connection

    # TODO: Implement performance monitoring, versioning?

    def __init__(self, rootPath: str | None = None) -> None:
        if rootPath is None:
            rootPath = DB_ROOT_PATH
        self.rootPath = rootPath
        self.modelDb = sqlite3.connect(f"{self.rootPath}/model.db")
        self.modelDb.execute(
            """CREATE TABLE IF NOT EXISTS models (
                model_type TEXT,
                model_tag TEXT,
                model_location TEXT,
                num_epochs_trained INTEGER,
                avg_epoch_length REAL,
                avg_epoch_score REAL,
                PRIMARY KEY(model_type, model_tag)
            );"""
        )
        self.modelDb.commit()

    def shutDown(self) -> None:
        self.modelDb.close()

    def publishModel(self, model: DeepQNetwork, modelDbKey: ModelDbKey) -> None:
        r"""Push new model to DB. Determines location automatically. Epoch metrics are filled in with zeros."""
        modelLocation = self._generateLocation(modelDbKey)
        epochMetrics = EpochMetrics(numEpochsTrained=0, avgEpochLength=0.0, avgEpochScore=0.0)

        torch.save(model.state_dict(), modelLocation)
        entry = ModelDbEntry(modelDbKey=modelDbKey, modelLocation=modelLocation, onlinePerformance=epochMetrics)
        self._upsert(entry)

    def updateModelMetrics(self, modelDbKey: ModelDbKey, metrics: EpochMetrics) -> None:
        r"""Push updated metrics to DB. Determines location automatically."""
        existingEntry = self._fetchEntry(modelDbKey)
        newMetrics = existingEntry.onlinePerformance.avgWith(metrics)
        entry = ModelDbEntry(
            modelDbKey=modelDbKey, modelLocation=existingEntry.modelLocation, onlinePerformance=newMetrics
        )
        self._upsert(entry)

    def pushBestModel(self) -> None:
        res = self.modelDb.execute("SELECT * FROM models ORDER BY avg_epoch_length DESC;").fetchone()
        entry = ModelDbEntry.fromDbRow(res)
        os.system(f"cp -f {entry.modelLocation} {MODEL_ROOT_PATH}")

    def _fetchEntry(self, modelDbKey: ModelDbKey) -> ModelDbEntry:
        res = self.modelDb.execute(
            f"""SELECT * FROM models
            WHERE model_type = '{modelDbKey.modelType.value}' AND model_tag = '{modelDbKey.modelTag}';"""
        ).fetchone()

        if res is None:
            raise RuntimeError(
                f"No entry found for model_type = {modelDbKey.modelType} and model_tag = {modelDbKey.modelTag}"
            )

        return ModelDbEntry.fromDbRow(res)

    def _upsert(self, entry: ModelDbEntry) -> None:
        self.modelDb.execute(
            f"""INSERT INTO models (
                model_type,
                model_tag,
                model_location,
                num_epochs_trained,
                avg_epoch_length,
                avg_epoch_score
            ) VALUES(
                '{entry.modelDbKey.modelType}',
                '{entry.modelDbKey.modelTag}',
                '{entry.modelLocation}',
                '{entry.onlinePerformance.numEpochsTrained}',
                '{entry.onlinePerformance.avgEpochLength}',
                '{entry.onlinePerformance.avgEpochScore}'
            )
            ON CONFLICT(model_type, model_tag)
            DO UPDATE SET
                model_location=excluded.model_location,
                num_epochs_trained=excluded.num_epochs_trained,
                avg_epoch_length=excluded.avg_epoch_length,
                avg_epoch_score=excluded.avg_epoch_score;"""
        )
        self.modelDb.commit()

    def _generateLocation(self, modelDbKey: ModelDbKey) -> str:
        directory = f"{self.rootPath}/models/{modelDbKey.modelType}/{modelDbKey.modelTag}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        return f"{directory}/weights.pt"
