from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.transition import Action, Context, Transition

C = TypeVar("C", bound=Context)
A = TypeVar("A", bound=Action)


class Event(SerializableDataClass, Generic[C, A]):
    r"""
    Dataclass representing a generic history event, which is defined as a transition and
    an optimal action for that transition.
    """

    transition: Transition[C, A]
    optimalAction: A

    def actionWasHit(self) -> bool:
        return self.transition.action == self.optimalAction


class History(SerializableDataClass, Generic[C, A]):
    r"""
    Generic history class which represents a history of interactions between a bandit
    agent and its environment.  Contains methods for retrieving the list of rewards and
    optimal action hits.
    """

    historyVector: list[Event]

    def __init__(self) -> None:
        self.historyVector = []

    def update(self, transition: Transition[C, A], optimalAction: A) -> None:
        self.historyVector.append(
            Event(transition=transition, optimalAction=optimalAction)
        )

    def getRewardsVector(self) -> NDArray[np.float64]:
        return np.array([event.transition.reward for event in self.historyVector])

    def getOptimalActionHitsVector(self) -> NDArray[np.bool]:
        return np.array([event.actionWasHit() for event in self.historyVector])
