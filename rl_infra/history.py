from abc import ABC
from typing import Generic, TypeVar

from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.base_types import SerializableDataClass
from rl_infra.transition import Action, Context, Transition

C = TypeVar("C", bound=Context, covariant=True)
A = TypeVar("A", bound=Action, covariant=True)


class History(ABC, SerializableDataClass, Generic[C, A]):
    transition_history: list[Transition[C, A]]

    def update(self, transition: Transition[C, A]) -> None: ...

    def getRewardsVector(self) -> NDArray[float64]: ...

    def getOptimalActionHitsVector(self) -> NDArray[bool]: ...
