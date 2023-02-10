from __future__ import annotations

import random

from rl_infra.impl.tetris.offline.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisEpochRecord, TetrisGameplayRecord, TetrisOnlineMetrics
from rl_infra.impl.tetris.online.tetris_transition import TetrisAction, TetrisState, TetrisTransition
from rl_infra.types.offline import DataService, SqliteConnection
from rl_infra.types.online.transition import DataDbRow


class TetrisDataService(DataService[TetrisState, TetrisAction, TetrisTransition, TetrisOnlineMetrics]):
    dbPath: str
    capacity: int

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

    def pushGameplay(self, gameplay: TetrisGameplayRecord) -> None:
        for epoch in gameplay.epochs:
            self.pushEpoch(epoch)

    def pushEpoch(self, epoch: TetrisEpochRecord) -> None:
        entries = list(
            map(
                lambda move: TetrisTransition(**move.dict()),
                epoch.moves,
            )
        )
        return self.pushEntries(entries)

    def pushEntries(self, entries: list[TetrisTransition]) -> None:
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

    def sample(self, batchSize: int) -> list[TetrisTransition]:
        with SqliteConnection(self.dbPath) as cur:
            numZeros = cur.execute(f"select count(*) from data where reward=0 limit {self.capacity}").fetchone()[0]
            numNonZeros = cur.execute("select count(*) from data where reward<>0").fetchone()[0]
            ratio = numZeros // numNonZeros
            allRows = cur.execute(
                f"""with RECURSIVE cte(x) AS (
                    SELECT 1 UNION ALL SELECT x + 1 FROM cte WHERE x < {ratio}
                ),
                nonzeros as (
                    select * from data where reward <> 0
                ),
                nonzeros_repeated as (
                    select nz.* from nonzeros nz cross join cte c
                ),
                zeros as (
                    select * from data where reward = 0 order by rowid desc limit {self.capacity}
                )
                select * from (
                    select * from nonzeros_repeated union all select * from zeros
                ) order by random() limit {batchSize}"""
            ).fetchall()
        if len(allRows) < batchSize:
            raise ValueError("Not enough results for batch.  Reduce batch size or collect more data.")
        return [TetrisTransition.from_orm(DataDbRow(*row)) for row in random.sample(allRows, batchSize)]