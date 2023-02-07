from abc import ABC
from typing import Generic, TypeVar

from rl_infra.types.base_types import SerializableDataClass


# States will vary quite a bit between implementations, so I am just using this class as a type stub.
class State(SerializableDataClass):
    pass


Action = str
S = TypeVar("S", bound=State, covariant=False, contravariant=False)
A = TypeVar("A", bound=Action, covariant=False, contravariant=False)


# This is really an interface, but I have to use ABC here because pydantic does not support mixing in with protocols.
class Transition(ABC, SerializableDataClass, Generic[S, A]):
    state: S
    action: A
    newState: S
    reward: float
    isTerminal: bool