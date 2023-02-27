from __future__ import annotations

import random
from typing import Sequence

from rl_infra.impl.tetris.offline.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisOnlineMetrics
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState, TetrisTransition
from rl_infra.types.offline import DataService, SqliteConnection
from rl_infra.types.online.environment import EpochRecord, GameplayRecord
from rl_infra.types.online.transition import DataDbRow, Transition


class TetrisDataService(DataService[TetrisState, TetrisAction, TetrisOnlineMetrics]):
    dbPath: str

    def __init__(self, rootPath: str | None = None, capacity: int | None = None) -> None:
        if rootPath is None:
            rootPath = DB_ROOT_PATH
        self.dbPath = f"{rootPath}/data.db"
        if capacity is None:
            capacity = 10000
        self.capacity = capacity
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS data (
                    state TEXT,
                    action TEXT,
                    new_state TEXT,
                    reward REAL,
                    is_terminal BOOLEAN
                );"""
            )

    def pushGameplay(self, gameplay: GameplayRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]) -> None:
        for epoch in gameplay.epochs:
            self.pushEpoch(epoch)

    def pushEpoch(self, epoch: EpochRecord[TetrisState, TetrisAction, TetrisOnlineMetrics]) -> None:
        return self.pushEntries(epoch.moves)

    def pushEntries(self, entries: Sequence[Transition[TetrisState, TetrisAction]]) -> None:
        query = """
            INSERT INTO data (
                state,
                action,
                new_state,
                reward,
                is_terminal
            ) VALUES (?, ?, ?, ?, ?);"""
        values = [entry.toDbRow() for entry in entries]
        with SqliteConnection(self.dbPath) as cur:
            cur.executemany(query, values)

    def sample(self, batchSize: int) -> Sequence[Transition[TetrisState, TetrisAction]]:
        with SqliteConnection(self.dbPath) as cur:
            numPositive = cur.execute("select count(*) from data where reward > 0").fetchone()[0]
            numZeros = cur.execute(f"select count(*) from data where reward = 0 limit {self.capacity}").fetchone()[0]
            numNegative = cur.execute("select count(*) from data where reward < 0").fetchone()[0]
            try:
                posRatio = numZeros // numPositive
                negRatio = numZeros // numNegative
            except ZeroDivisionError:
                raise ValueError(
                    "No results with either positive or negative reward.  Collect more data before training."
                )
            sampleSize = batchSize // 3
            remainder = batchSize % 3
            posRows = cur.execute(
                f"""
                with recursive cte(x) AS (
                    select 1 union all select x + 1 from cte where x < {posRatio}
                ),
                positives as (
                    select * from data where reward > 0
                ),
                positives_repeated as (
                    select p.* from positives p cross join cte c
                )
                select * from positives_repeated order by random() limit {sampleSize + remainder};
                """
            ).fetchall()
            negRows = cur.execute(
                f"""
                with recursive cte(x) AS (
                    select 1 union all select x + 1 from cte where x < {negRatio}
                ),
                negatives as (
                    select * from data where reward < 0
                ),
                negatives_repeated as (
                    select n.* from negatives n cross join cte c
                )
                select * from negatives_repeated order by random() limit {sampleSize};
                """
            ).fetchall()
            zeroRows = cur.execute(
                f"""
                with zeros as (
                    select * from data where reward = 0 order by rowid desc limit {self.capacity}
                )
                select * from zeros order by random() limit {sampleSize};
                """
            ).fetchall()
        allRows = posRows + negRows + zeroRows
        if len(allRows) < batchSize:
            raise ValueError("Not enough results for batch.  Reduce batch size or collect more data.")
        return [TetrisTransition.from_orm(DataDbRow(*row)) for row in random.sample(allRows, batchSize)]

    def keepNewRowsDeleteOld(self, sgn: int = 0, numToKeep: int = 1000) -> None:
        if sgn not in [-1, 0, 1]:
            raise KeyError("sgn must be one of {-1, 0, 1}")
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                f"""
                with rows_to_keep as (
                    select rowid from data
                    where sign(reward) = {sgn}
                    order by rowid desc
                    limit {numToKeep}
                )
                delete from data where sign(reward) = {sgn} and rowid not in rows_to_keep;
                """
            )
