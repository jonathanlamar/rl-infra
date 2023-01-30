from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Optional, Tuple, Type

import torch

from ...tetris.models.dqn import DeepQNetwork


@dataclass
class ModelType(str, Enum):
    ACTOR = "ACTOR"
    CRITIC = "CRITIC"


@dataclass
class EpochMetrics:
    avgEpochLength: float
    avgEpochScore: float

    def avgWith(self, other: EpochMetrics) -> EpochMetrics:
        return EpochMetrics(
            avgEpochLength=(self.avgEpochLength + other.avgEpochLength) / 2.0,
            avgEpochScore=(self.avgEpochScore + other.avgEpochScore) / 2.0,
        )


DbRow = Tuple[str, str, str, float, float]


@dataclass
class ModelDbEntry:
    modelType: ModelType
    modelTag: str
    modelLocation: str
    onlinePerformance: EpochMetrics

    def valueExpr(self) -> str:
        return f"""
            VALUES(
                '{self.modelType.value}',
                '{self.modelTag}',
                '{self.modelLocation}',
                '{self.onlinePerformance.avgEpochLength}',
                '{self.onlinePerformance.avgEpochScore}'
            )
        """

    @staticmethod
    def fromDbRow(row: DbRow) -> ModelDbEntry:
        return ModelDbEntry(
            modelType=ModelType[row[0]],
            modelTag=row[1],
            modelLocation=row[2],
            onlinePerformance=EpochMetrics(avgEpochLength=row[3], avgEpochScore=row[4]),
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

    def publishModel(self, model: DeepQNetwork, modelType: ModelType, modelTag: str) -> None:
        r"""Push new model to DB. Determines location automatically. Epoch metrics are filled in with zeros."""
        modelLocation = self._generateLocation(modelTag, modelType)
        epochMetrics = EpochMetrics(0.0, 0.0)

        torch.save(model.state_dict(), modelLocation)
        entry = ModelDbEntry(modelType, modelTag, modelLocation, epochMetrics)
        self._upsert(entry)
        match modelType:
            case ModelType.ACTOR:
                self.latestActorModelEntry = entry
            case ModelType.CRITIC:
                self.latestCriticModelEntry = entry

    def updateModelMetrics(self, modelType: ModelType, modelTag: str, metrics: EpochMetrics) -> None:
        r"""Push updated metrics to DB. Determines location automatically."""
        existingEntry = self._fetchEntry(modelType, modelTag)
        newMetrics = existingEntry.onlinePerformance.avgWith(metrics)
        entry = ModelDbEntry(modelType, modelTag, existingEntry.modelLocation, newMetrics)
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

    def _fetchEntry(self, modelType: ModelType, modelTag: str) -> ModelDbEntry:
        res = self.modelDb.execute(
            f"""SELECT * FROM models WHERE model_type = '{modelType.value}' AND model_tag = '{modelTag}';"""
        ).fetchone()

        if res is None:
            raise RuntimeError(f"No entry found for model_type = {modelType} and model_tag = {modelTag}")

        return ModelDbEntry.fromDbRow(res)

    def _upsert(self, entry: ModelDbEntry) -> None:
        self.modelDb.execute(
            f"""INSERT INTO models (
                model_type,
                model_tag,
                model_location,
                avg_epoch_length,
                avg_epoch_score
            ) {entry.valueExpr()}
            ON CONFLICT(model_type, model_tag)
            DO UPDATE SET
                model_location=excluded.model_location,
                avg_epoch_length=excluded.avg_epoch_length,
                avg_epoch_score=excluded.avg_epoch_score;"""
        )

    def _generateLocation(self, modelTag: str, modelType: ModelType) -> str:
        return f"{self.rootPath}/{modelType}/{modelTag}/model.pt"
