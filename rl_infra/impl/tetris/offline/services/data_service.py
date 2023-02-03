from __future__ import annotations

from rl_infra.impl.tetris.offline.services.config import DB_ROOT_PATH
from rl_infra.impl.tetris.online.tetris_environment import TetrisTransition
from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.offline.services import DataService, DbRow, SqliteConnection

DataDbRow = DbRow[str, int, int]


class DataDbEntry(SerializableDataClass):
    transition: TetrisTransition
    epoch: int
    move: int

    @staticmethod
    def fromDbRow(row: DataDbRow) -> DataDbEntry:
        # TODO: This might be doable from pydantic builtins
        return DataDbEntry(transition=TetrisTransition.parse_raw(row[0]), epoch=row[1], move=row[2])


class TetrisDataService(DataService[DataDbEntry]):
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

    def push(self, entries: list[DataDbEntry]) -> None:
        query = "INSERT INTO data (transition, epoch_num, move_num) VALUES (?, ?, ?);"
        values = [(entry.transition.json(), entry.epoch, entry.move) for entry in entries]
        with SqliteConnection(self.dbPath) as cur:
            cur.executemany(query, values)

    def sample(self, batchSize: int) -> list[DataDbEntry]:
        # FIXME: This needs to be a rencency-weighted random selection.
        with SqliteConnection(self.dbPath) as cur:
            rows = cur.execute("SELECT * FROM data ORDER BY epoch_num DESC, move_num DESC").fetchmany(batchSize)
        return [DataDbEntry.fromDbRow(row) for row in rows]
