from typing import Protocol, Sequence, TypeVar

import numpy as np
from numpy import float64
from numpy.typing import NDArray

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.bandit_problem import StationaryBanditProblem
from rl_infra.types.testbed import TestBed
from rl_infra.types.transition import Action, Context

Ag = TypeVar("Ag", bound=StationaryBanditAgent)


class StationaryBanditTestBed(TestBed[Context, Action, Ag], Protocol[Ag]):
    bandits: Sequence[StationaryBanditProblem[Ag]]

    def play(self, num_rounds: int) -> None:
        for bandit in self.bandits:
            bandit.play(num_rounds)

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
