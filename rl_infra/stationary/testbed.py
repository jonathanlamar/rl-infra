from typing import Generic, Sequence, Type, TypeVar

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.bandit_problem import StationaryBanditProblem
from rl_infra.types.testbed import TestBed
from rl_infra.types.transition import Action, Context

Ag = TypeVar("Ag", bound=StationaryBanditAgent)


class StationaryBanditTestBed(TestBed[Context, Action, Ag], Generic[Ag]):
    r"""
    A testbed of stationary bandits.  This is a list of stationary bandit problems along
    with a method for making all of them play.
    """

    bandits: Sequence[StationaryBanditProblem[Ag]]

    def __init__(self, agentClass: Type[Ag], numBandits: int, numArms: int) -> None:
        self.bandits = [  # pyright: ignore
            StationaryBanditProblem(agentClass, numArms)  # pyright: ignore
            for _ in range(numBandits)
        ]

    def play(self, numRounds: int) -> None:
        for bandit in self.bandits:
            bandit.play(numRounds)

    def getAverageRewardsVector(self) -> NDArray[float64]:
        return np.concat(
            [bandit.getRewardsVector().reshape(1, -1) for bandit in self.bandits]
        ).mean(axis=1)

    def getPercentOptimalActionVector(self) -> NDArray[float64]:
        return np.concat(
            [
                bandit.getOptimalActionHitsVector().reshape(1, -1)
                for bandit in self.bandits
            ]
        ).mean(axis=0)
