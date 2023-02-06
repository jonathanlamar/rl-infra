from __future__ import annotations

from rl_infra.impl.tetris.offline.services.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_environment import (
    TetrisAction,
    TetrisEpochRecord,
    TetrisOnlineMetrics,
    TetrisGameplayRecord,
    TetrisState,
    TetrisTransition,
)
from rl_infra.types.offline import DataDbEntry, DataDbRow, DataService, SqliteConnection


class TetrisDataDbEntry(DataDbEntry[TetrisTransition]):
    transition: TetrisTransition
    epoch: int
    move: int

    @staticmethod
    def fromDbRow(row: DataDbRow) -> TetrisDataDbEntry:
        # TODO: This might be doable from pydantic builtins
        return TetrisDataDbEntry(transition=TetrisTransition.parse_raw(row[0]), epoch=row[1], move=row[2])


class TetrisDataService(
    DataService[TetrisDataDbEntry, TetrisState, TetrisAction, TetrisTransition, TetrisOnlineMetrics]
):
    dbPath: str

    def __init__(self, rootPath: str | None = None) -> None:
        if rootPath is None:
            rootPath = DB_ROOT_PATH
        self.dbPath = f"{rootPath}/data.db"
        with SqliteConnection(self.dbPath) as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS data (
                    transition TEXT,
                    epoch_num INTEGER,
                    move_num INTEGER
                );"""
            )

    def pushGameplay(self, gameplay: TetrisGameplayRecord) -> None:
        for epoch in gameplay.epochs:
            self.pushEpoch(epoch)

    def pushEpoch(self, epoch: TetrisEpochRecord) -> None:
        epochNumber = epoch.epochNumber
        entries = list(
            map(
                lambda tup: TetrisDataDbEntry(transition=tup[1], epoch=epochNumber, move=tup[0]), enumerate(epoch.moves)
            )
        )
        return self.pushEntries(entries)

    def pushEntries(self, entries: list[TetrisDataDbEntry]) -> None:
        query = "INSERT INTO data (transition, epoch_num, move_num) VALUES (?, ?, ?);"
        values = [(entry.transition.json(), entry.epoch, entry.move) for entry in entries]
        with SqliteConnection(self.dbPath) as cur:
            cur.executemany(query, values)

    def sample(self, batchSize: int) -> list[TetrisDataDbEntry]:
        # FIXME: This needs to be a rencency-weighted random selection.
        with SqliteConnection(self.dbPath) as cur:
            rows = cur.execute("SELECT * FROM data ORDER BY epoch_num DESC, move_num DESC").fetchmany(batchSize)
        return [TetrisDataDbEntry.fromDbRow(row) for row in rows]
