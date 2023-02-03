from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from types import TracebackType
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


class SqliteConnection(AbstractContextManager[sqlite3.Cursor]):
    dbPath: str
    connection: sqlite3.Connection | None

    def __init__(self, dbPath: str) -> None:
        self.dbPath = dbPath

    def __enter__(self) -> sqlite3.Cursor:
        self.connection = sqlite3.connect(self.dbPath)
        return self.connection.cursor()

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        if self.connection is not None:
            self.connection.commit()
            self.connection.close()
        else:
            raise TypeError("connection is None type")
        return super().__exit__(__exc_type, __exc_value, __traceback)


class DataDbEntry(SerializableDataClass):
    transition: TetrisTransition
    epoch: int
    move: int

    @staticmethod
    def fromDbRow(row: DataDbRow) -> DataDbEntry:
        # TODO: This might be doable from pydantic builtins
        return DataDbEntry(transition=TetrisTransition.parse_raw(row[0]), epoch=row[1], move=row[2])


class DataService(ReplayMemory[TetrisTransition]):
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
