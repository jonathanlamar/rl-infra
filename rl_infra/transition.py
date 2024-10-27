from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from pydantic import BaseModel, field_validator
from typing_extensions import Self


class SerializableDataClass(BaseModel):
    class Config:
        allow_mutation = False
        use_enum_values = True
        orm_mode = True


# States will vary quite a bit between implementations, so I am just using this class as a type stub.
class State(SerializableDataClass):
    isTerminal: bool


Action = str
S = TypeVar("S", bound=State, covariant=True)
A = TypeVar("A", bound=Action, covariant=True)


# This is really an interface, but I have to use ABC here because pydantic does not support mixing in with protocols.
class Transition(ABC, SerializableDataClass, Generic[S, A]):
    state: S
    action: A
    newState: S
    reward: float

    @field_validator("state", "newState", mode="before")
    @classmethod
    @abstractmethod
    def _parseStateFromJson(cls: Type[Self], val: S | str) -> S: ...
