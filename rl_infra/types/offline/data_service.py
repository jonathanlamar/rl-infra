from abc import ABC
from typing import Generic, Protocol, TypeVar

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.online.environment import Transition

DataDbRow = tuple[str, int, int]
DbEntry = TypeVar("DbEntry", bound=SerializableDataClass, covariant=False, contravariant=False)
T = TypeVar("T", bound=Transition)


class DataDbEntry(ABC, SerializableDataClass, Generic[T]):
    transition: T
    epoch: int
    move: int


class DataService(Protocol[DbEntry]):
    def push(self, entries: list[DbEntry]) -> None:
        ...

    def sample(self, batchSize: int) -> list[DbEntry]:
        ...
