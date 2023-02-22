from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, Type, TypeVar

from pydantic import validator
from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass


# States will vary quite a bit between implementations, so I am just using this class as a type stub.
class State(SerializableDataClass):
    pass


Action = str
S = TypeVar("S", bound=State, covariant=True)
A = TypeVar("A", bound=Action, covariant=True)


class DataDbRow(NamedTuple):
    state: str
    action: str
    newState: str
    reward: float
    isTerminal: bool


# This is really an interface, but I have to use ABC here because pydantic does not support mixing in with protocols.
class Transition(ABC, SerializableDataClass, Generic[S, A]):
    state: S
    action: A
    newState: S
    reward: float
    isTerminal: bool

    @validator("state", "newState", pre=True)
    @classmethod
    @abstractmethod
    def _parseStateFromJson(cls: Type[Self], val: S | str) -> S:
        ...

    def toDbRow(self) -> DataDbRow:
        return DataDbRow(
            state=self.state.json(),
            action=self.action,
            newState=self.newState.json(),
            reward=self.reward,
            isTerminal=self.isTerminal,
        )
