from __future__ import annotations

import sqlite3
from typing import Protocol, TypeVar

from rl_infra.base_types import SerializableDataClass
from rl_infra.offline.tetris.services.config import DB_ROOT_PATH
from rl_infra.online.impl.tetris.tetris_environment import TetrisTransition
from rl_infra.online.types.environment import Transition

T = TypeVar("T", bound=Transition, covariant=False, contravariant=False)


# Not necessary for now, but I might do some abstraction once I get tetris running.
class ReplayMemory(Protocol[T]):
    def push(self, transition: T, epoch: int, move: int) -> None:
        ...

    def sample(self, batchSize: int) -> list[T]:
        ...


DataDbRow = tuple[str, int, int]


class DataDbEntry(SerializableDataClass):
    transition: TetrisTransition
    epoch: int
    move: int

    @staticmethod
    def fromDbRow(row: DataDbRow) -> DataDbEntry:
        # TODO: This might be doable from pydantic builtins
        return DataDbEntry(transition=TetrisTransition.parse_raw(row[0]), epoch=row[1], move=row[2])


class DataService(ReplayMemory[TetrisTransition]):
    rootPath: str
    dataDb: sqlite3.Connection

    def __init__(self, rootPath: str | None = None) -> None:
        if rootPath is None:
            rootPath = DB_ROOT_PATH
        self.rootPath = rootPath
        self.dataDb = sqlite3.connect(f"{rootPath}/data.db")
        self.dataDb.execute(
            """CREATE TABLE IF NOT EXISTS data (
                transition TEXT,
                epoch_num INTEGER,
                move_num INTEGER
            );"""
        )
        self.dataDb.commit()

    def shutDown(self) -> None:
        self.dataDb.close()

    def push(self, entries: list[DataDbEntry]) -> None:
        query = "INSERT INTO data (transition, epoch_num, move_num) VALUES (?, ?, ?);"
        values = [(entry.transition.json(), entry.epoch, entry.move) for entry in entries]
        self.dataDb.executemany(query, values)
        self.dataDb.commit()

    def sample(self, batchSize: int) -> list[DataDbEntry]:
        # FIXME: This needs to be a rencency-weighted random selection.
        rows = self.dataDb.execute("SELECT * FROM data ORDER BY epoch_num DESC, move_num DESC").fetchmany(batchSize)
        return [DataDbEntry.fromDbRow(row) for row in rows]
