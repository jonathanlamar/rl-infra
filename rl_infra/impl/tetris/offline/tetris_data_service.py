from __future__ import annotations

import logging
import random
from math import ceil
from typing import Sequence

from rl_infra.impl.tetris.offline.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_environment import (
    TetrisEpisodeRecord,
    TetrisGameplayRecord,
    TetrisOnlineMetrics,
)
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState, TetrisTransition
from rl_infra.types.offline import DataService, SqliteConnection
from rl_infra.types.online.environment import EpisodeRecord
from rl_infra.types.online.transition import DataDbRow, Transition

logger = logging.getLogger(__name__)


class TetrisDataService(DataService[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    dbPath: str

    def __init__(self, rootPath: str | None = None, capacity: int = 10000) -> None:
        if rootPath is None:
            rootPath = DB_ROOT_PATH
        self.dbPath = f"{rootPath}/data.db"
        self.capacity = capacity
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS data (
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    reward REAL NOT NULL
                );"""
            )
            cur.execute(
                """CREATE TABLE IF NOT EXISTS validation_data (
                    episode_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    reward REAL NOT NULL
                );"""
            )

    def pushGameplay(self, gameplay: TetrisGameplayRecord) -> None:
        logger.info("Pushing gameplay record")
        for episode in gameplay.episodes:
            self.pushEpisode(episode)

    def pushEpisode(self, episode: EpisodeRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]) -> None:
        logger.info("Pushing episode record.")
        logger.debug(f"Episode: {episode}")
        query = """
            INSERT INTO data (
                state,
                action,
                new_state,
                reward
            ) VALUES (?, ?, ?, ?);"""
        values = [entry.toDbRow() for entry in episode.moves]
        with SqliteConnection(self.dbPath) as cur:
            cur.executemany(query, values)

    def pushValidationEpisode(self, episode: EpisodeRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]) -> None:
        with SqliteConnection(self.dbPath) as cur:
            maxId = cur.execute("SELECT MAX(episode_id) FROM validation_data;").fetchone()[0]
        if maxId is None:
            id = 0
        else:
            id = maxId + 1
        logger.info("Pushing validation episode.")
        logger.info(f"Validation episode ID: {id}")
        logger.debug(f"Episode: {episode}")
        query = """
            INSERT INTO validation_data (
                episode_id,
                state,
                action,
                new_state,
                reward
            ) VALUES (?, ?, ?, ?, ?);"""
        values = [(id,) + entry.toDbRow() for entry in episode.moves]
        with SqliteConnection(self.dbPath) as cur:
            cur.executemany(query, values)

    def getValidationEpisode(
        self, episodeId: int | None = None
    ) -> EpisodeRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]:
        if episodeId is None:
            with SqliteConnection(self.dbPath) as cur:
                maxId = cur.execute("SELECT MAX(episode_id) FROM validation_data;").fetchone()[0]
            if maxId is None:
                raise KeyError("No validation episodes")
            else:
                episodeId = maxId
        logger.info(f"Retrieving validation episode with id = {episodeId}")
        with SqliteConnection(self.dbPath) as cur:
            rows = cur.execute(
                f"SELECT state, action, new_state, reward FROM validation_data WHERE episode_id = {episodeId}"
            ).fetchall()
        return TetrisEpisodeRecord(episodeNumber=0, moves=[TetrisTransition.from_orm(DataDbRow(*row)) for row in rows])

    def sample(self, batchSize: int) -> Sequence[Transition[TetrisState, TetrisAction]]:
        logger.info(f"Sampling batch of {batchSize} transitions")
        with SqliteConnection(self.dbPath) as cur:
            rows = cur.execute(f"select * from data order by random() limit {batchSize}").fetchall()
        if len(rows) < batchSize:
            logger.info(f"Not enough rows found (found {len(rows)}).  Oversampling.")
            rows *= ceil(batchSize / len(rows))
            random.shuffle(rows)
            rows = rows[:batchSize]
            logger.debug(f"Oversampled rows: {rows}")
        return [TetrisTransition.from_orm(DataDbRow(*row)) for row in random.sample(rows, batchSize)]

    def keepNewRowsDeleteOld(self, sgn: int = 0) -> None:
        logger.info(f"Removing all but {self.capacity} rows with reward sign {sgn}")
        if sgn not in [-1, 0, 1]:
            raise KeyError("sgn must be one of {-1, 0, 1}")
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""
                with rows_to_keep as (
                    select rowid from data
                    where sign(reward) = {sgn}
                    order by rowid desc
                    limit {self.capacity}
                )
                delete from data where sign(reward) = {sgn} and rowid not in rows_to_keep;
                """
            )
