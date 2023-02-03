import sqlite3
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol, TypeVar

from rl_infra.types.base_types import SerializableDataClass

DbRow = tuple

DbEntry = TypeVar("DbEntry", bound=SerializableDataClass, covariant=False, contravariant=False)


class DataService(Protocol[DbEntry]):
    def push(self, entries: list[DbEntry]) -> None:
        ...

    def sample(self, batchSize: int) -> list[DbEntry]:
        ...


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
