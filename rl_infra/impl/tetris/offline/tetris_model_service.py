from __future__ import annotations

import os

import torch
from torch.optim import Optimizer

from rl_infra.impl.tetris.offline.config import DB_ROOT_PATH
from rl_infra.impl.tetris.offline.dqn import DeepQNetwork
from rl_infra.impl.tetris.offline.tetris_schema import (
    TetrisModelDbEntry,
    TetrisModelDbRow,
    TetrisOfflineMetrics,
    TetrisOfflineMetricsDbEntry,
    TetrisOnlineMetricsDbEntry,
)
from rl_infra.impl.tetris.online.config import MODEL_ENTRY_PATH, MODEL_ROOT_PATH, MODEL_WEIGHTS_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisOnlineMetrics
from rl_infra.types.offline import ModelDbKey, ModelService, SqliteConnection


class TetrisModelService(ModelService[DeepQNetwork]):
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
                    weights_location TEXT NOT NULL,
                    num_episodes_played INTEGER NOT NULL,
                    num_epochs_trained INTEGER NOT NULL,
                    avg_episode_length REAL,
                    avg_episode_score REAL,
                    recency_weighted_avg_loss REAL,
                    recency_weighted_avg_validation_q REAL,
                    PRIMARY KEY(tag, version)
                );"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS offline_metrics (
                    tag TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    epoch_number INTEGER NOT NULL,
                    num_batches_trained INTEGER NOT NULL
                    avg_batch_loss REAL NOT NULL,
                    val_episode_avg_max_q REAL NOT NULL,
                    validation_episode_id INTEGER NOT NULL
                    FOREIGN KEY(tag, version) REFERENCES models(tag, version)
                );"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS online_metrics (
                    tag TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    episode_number INTEGER NOT NULL,
                    num_moves INTEGER NOT NULL
                    score INTEGER NOT NULL,
                    FOREIGN KEY(tag, version) REFERENCES models(tag, version)
                );"""
            )

    def publishNewModel(
        self,
        modelTag: str,
        policyModel: DeepQNetwork | None = None,
        targetModel: DeepQNetwork | None = None,
        optimizer: Optimizer | None = None,
    ) -> int:
        maybeExistingModelKey = self.getLatestVersionKey(modelTag)
        if maybeExistingModelKey is not None:
            version = maybeExistingModelKey.version + 1
        else:
            version = 0
        weightsLocation = self._generateWeightsLocation(modelTag, version)
        newModelKey = ModelDbKey(tag=modelTag, version=version, weightsLocation=weightsLocation)
        self.updateModel(newModelKey, policyModel=policyModel, targetModel=targetModel, optimizer=optimizer)
        return version

    def getModelKey(self, modelTag: str, version: int) -> ModelDbKey:
        weightsLocation = self._generateWeightsLocation(modelTag, version)
        return ModelDbKey(tag=modelTag, version=version, weightsLocation=weightsLocation)

    def getLatestVersionKey(self, modelTag: str) -> ModelDbKey | None:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute(
                f"SELECT tag, version, weights_location FROM models WHERE tag = '{modelTag}' ORDER BY version DESC"
            ).fetchone()
        if res is None:
            return None
        return ModelDbKey(tag=res[0], version=res[1], weightsLocation=res[2])

    def getModelEntry(self, key: ModelDbKey) -> TetrisModelDbEntry | None:
        with SqliteConnection(self.dbPath) as cur:
            res = cur.execute(
                f"""SELECT * FROM models WHERE tag = '{key.tag}' AND version = {key.version};"""
            ).fetchone()
        if res is None:
            return None
        return TetrisModelDbEntry.from_orm(TetrisModelDbRow(*res))

    def deployModel(self, key: ModelDbKey) -> None:
        entry = self.getModelEntry(key)
        if entry is None:
            raise KeyError(f"Model {key} not found")
        if not os.path.exists(self.deployModelRootPath):
            os.makedirs(self.deployModelRootPath)
        os.system(f"cp {key.policyModelLocation} {self.deployModelWeightsPath}")
        with open(self.deployModelEntryPath, "w") as f:
            f.write(entry.json())

    def updateModel(
        self,
        key: ModelDbKey,
        policyModel: DeepQNetwork | None = None,
        targetModel: DeepQNetwork | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        modelEntry = self.getModelEntry(key)
        if modelEntry is None:
            raise KeyError(f"Model {key} not found")
        maybeExistingEntry = self.getModelEntry(key)
        if maybeExistingEntry is not None:
            modelEntry = maybeExistingEntry.updateWithNewValues(modelEntry)
        self._writeModelWeights(key, policyModel, targetModel, optimizer)
        self._upsertModelEntry(modelEntry)

    def publishOnlineMetrics(self, key: ModelDbKey, onlineMetrics: TetrisOnlineMetrics) -> None:
        modelEntry = TetrisModelDbEntry.fromMetrics(key, onlineMetrics=onlineMetrics)
        maybeExistingEntry = self.getModelEntry(key)
        if maybeExistingEntry is not None:
            modelEntry = maybeExistingEntry.updateWithNewValues(modelEntry)
        self._upsertModelEntry(modelEntry)
        onlineMetricsEntry = TetrisOnlineMetricsDbEntry(modelDbKey=key, onlineMetrics=onlineMetrics)
        self._insertOnlineMetricsEntry(onlineMetricsEntry)

    def publishOfflineMetrics(self, key: ModelDbKey, offlineMetrics: TetrisOfflineMetrics) -> None:
        modelEntry = TetrisModelDbEntry.fromMetrics(key, offlineMetrics=offlineMetrics)
        maybeExistingEntry = self.getModelEntry(key)
        if maybeExistingEntry is not None:
            modelEntry = maybeExistingEntry.updateWithNewValues(modelEntry)
        self._upsertModelEntry(modelEntry)
        offlineMetricsEntry = TetrisOfflineMetricsDbEntry(modelDbKey=key, offlineMetrics=offlineMetrics)
        self._insertOfflineMetricsEntry(offlineMetricsEntry)

    def _insertOnlineMetricsEntry(self, entry: TetrisOnlineMetricsDbEntry) -> None:
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""INSERT INTO online_metrics (
                    tag,
                    version,
                    episode_number,
                    num_moves,
                    score
                ) VALUES (
                    '{entry.modelDbKey.tag}',
                    '{entry.modelDbKey.version}',
                    {entry.onlineMetrics.episodeNumber},
                    {entry.onlineMetrics.numMoves},
                    {entry.onlineMetrics.score}
                )
                ON CONFLICT (tag, version)
                DO UPDATE SET
                    episode_number=excluded.episode_number,
                    num_moves=excluded.num_moves,
                    score=excluded.score,
                """
            )

    def _insertOfflineMetricsEntry(self, entry: TetrisOfflineMetricsDbEntry) -> None:
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""INSERT INTO offline_metrics (
                    tag,
                    version,
                    epoch_number,
                    num_batches_trained,
                    avg_batch_loss,
                    val_episode_avg_max_q,
                    validation_episode_id
                ) VALUES (
                    '{entry.modelDbKey.tag}',
                    '{entry.modelDbKey.version}',
                    {entry.offlineMetrics.epochNumber},
                    {entry.offlineMetrics.numBatchesTrained}
                    {entry.offlineMetrics.avgBatchLoss},
                    {entry.offlineMetrics.valEpisodeAvgMaxQ},
                    {entry.offlineMetrics.validationEpisodeId}
                )
                ON CONFLICT (tag, version)
                DO UPDATE SET
                    epoch_number=exluded.epoch_number,
                    num_batches_trained=exluded.num_batches_trained,
                    avg_batch_loss=exluded.avg_batch_loss,
                    val_episode_avg_max_q=exluded.val_episode_avg_max_q,
                    validation_episode_id=excluded.validation_episode_id
                """
            )

    def _upsertModelEntry(self, entry: TetrisModelDbEntry) -> None:
        episodeLength = entry.avgEpisodeLength if entry.avgEpisodeLength is not None else "NULL"
        episodeScore = entry.avgEpisodeScore if entry.avgEpisodeScore is not None else "NULL"
        trainingLoss = entry.recencyWeightedAvgLoss if entry.recencyWeightedAvgLoss is not None else "NULL"
        avgQ = entry.recencyWeightedAvgValidationQ if entry.recencyWeightedAvgValidationQ is not None else "NULL"
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""INSERT INTO models (
                    tag,
                    version,
                    weights_location,
                    num_episodes_played,
                    num_epochs_trained,
                    avg_episode_length,
                    avg_episode_score,
                    recency_weighted_avg_loss,
                    recency_weighted_avg_validation_q
                ) VALUES (
                    '{entry.modelDbKey.tag}',
                    '{entry.modelDbKey.version}',
                    '{entry.modelDbKey.weightsLocation}',
                    {entry.numEpisodesPlayed},
                    {entry.numEpochsTrained},
                    {episodeLength},
                    {episodeScore},
                    {trainingLoss},
                    {avgQ}
                )
                ON CONFLICT (tag, version)
                DO UPDATE SET
                    weights_location=excluded.weights_location,
                    num_episodes_played=excluded.num_episodes_played,
                    num_epochs_trained=excluded.num_epochs_trained,
                    avg_episode_length=excluded.avg_episode_length,
                    avg_episode_score=excluded.avg_episode_score,
                    recency_weighted_avg_loss=excluded.recency_weighted_avg_loss,
                    recency_weighted_avg_validation_q=excluded.recency_weighted_avg_validation_q;"""
            )

    def _writeModelWeights(
        self,
        key: ModelDbKey,
        policyModel: DeepQNetwork | None = None,
        targetModel: DeepQNetwork | None = None,
        optimizer: Optimizer | None = None,
    ) -> None:
        if not os.path.exists(key.weightsLocation):
            os.makedirs(key.weightsLocation)
        if policyModel is not None:
            torch.save(policyModel.state_dict(), key.policyModelLocation)
        if targetModel is not None:
            torch.save(targetModel.state_dict(), key.targetModelLocation)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), key.optimizerLocation)

    def _generateWeightsLocation(self, tag: str, version: int) -> str:
        return f"{self.modelWeightsPathStub}/{tag}/{version}"
