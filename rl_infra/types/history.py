from abc import ABC
from typing import Generic, TypeVar

import numpy as np
from numpy import bool, float64
from numpy.typing import NDArray

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.transition import Action, Context, Transition

C = TypeVar("C", bound=Context, covariant=True)
A = TypeVar("A", bound=Action, covariant=True)


class History(SerializableDataClass, Generic[C, A]):
    transitionHistory: list[Transition[C, A]]

    def __init__(self) -> None:
        self.transitionHistory = []

    def update(self, transition: Transition[C, A]) -> None:
        self.transitionHistory.append(transition)

    def getRewardsVector(self) -> NDArray[float64]:
        return np.array([transition.reward for transition in self.transitionHistory])

    def getOptimalActionHitsVector(self) -> NDArray[bool]:
        return np.array(
            [
                transition.action == transition.optimalAction
                for transition in self.transitionHistory
            ]
        )
