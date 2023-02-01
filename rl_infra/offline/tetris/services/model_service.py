from __future__ import annotations

import sqlite3
from enum import Enum
from types import TracebackType
from typing import Optional, Tuple, Type

import torch

from ....base_types import SerializableDataClass
from ...tetris.models.dqn import DeepQNetwork


class ModelType(str, Enum):
    ACTOR = "ACTOR"
    CRITIC = "CRITIC"


ModelDbRow = Tuple[str, str, str, int, float, float]


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
        return ModelDbEntry(
            modelDbKey=ModelDbKey(modelType=ModelType[row[0]], modelTag=row[1]),
            modelLocation=row[2],
            onlinePerformance=EpochMetrics(numEpochsTrained=row[3], avgEpochLength=row[4], avgEpochScore=row[5]),
        )


class ModelService:
    rootPath: str
    modelDb: sqlite3.Connection
    letestActorModelEntry: Optional[ModelDbEntry]
    letestCriticModelEntry: Optional[ModelDbEntry]

    # TODO: Implement performance monitoring, versioning?

    def __init__(self, rootPath: str) -> None:
        self.rootPath = rootPath
        self.modelDb = sqlite3.connect(f"{rootPath}/models.db")
        self.latestActorModelEntry = None
        self.latestCriticModelEntry = None

    def __enter__(self) -> ModelService:
        self.modelDb = sqlite3.connect(f"{self.rootPath}/models.db")
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
        return self

    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], traceback: Optional[TracebackType]
    ) -> None:
        self.modelDb.commit()
        self.modelDb.close()

    def publishModel(self, model: DeepQNetwork, modelDbKey: ModelDbKey) -> None:
        r"""Push new model to DB. Determines location automatically. Epoch metrics are filled in with zeros."""
        modelLocation = self._generateLocation(modelDbKey)
        epochMetrics = EpochMetrics(0, 0.0, 0.0)

        torch.save(model.state_dict(), modelLocation)
        entry = ModelDbEntry(modelDbKey=modelDbKey, modelLocation=modelLocation, onlinePerformance=epochMetrics)
        self._upsert(entry)
        match modelDbKey.modelType:
            case ModelType.ACTOR:
                self.latestActorModelEntry = entry
            case ModelType.CRITIC:
                self.latestCriticModelEntry = entry

    def updateModelMetrics(self, modelDbKey: ModelDbKey, metrics: EpochMetrics) -> None:
        r"""Push updated metrics to DB. Determines location automatically."""
        existingEntry = self._fetchEntry(modelDbKey)
        newMetrics = existingEntry.onlinePerformance.avgWith(metrics)
        entry = ModelDbEntry(
            modelDbKey=modelDbKey, modelLocation=existingEntry.modelLocation, onlinePerformance=newMetrics
        )
        self._upsert(entry)

    def getLatestModel(self, modelType: ModelType) -> DeepQNetwork:
        entry: Optional[ModelDbEntry]
        match modelType:
            case ModelType.ACTOR:
                if self.latestActorModelEntry is None:
                    raise RuntimeError("Latest actor model is undefined")
                entry = self.latestActorModelEntry
            case ModelType.CRITIC:
                if self.latestCriticModelEntry is None:
                    raise RuntimeError("Latest critic model is undefined")
                entry = self.latestCriticModelEntry

        return torch.load(entry.modelLocation)

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
                '{entry.modelDbKey.modelType.value}',
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

    def _generateLocation(self, modelDbKey: ModelDbKey) -> str:
        return f"{self.rootPath}/{modelDbKey.modelType}/{modelDbKey.modelTag}/model.pt"
