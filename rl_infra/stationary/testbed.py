from typing import Generic, Sequence, Type, TypeVar

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

    def __init__(self, agentClass: Type[Ag], numBandits: int, **kwargs) -> None:
        self.bandits = [  # pyright: ignore
            StationaryBanditProblem(agentClass, **kwargs) for _ in range(numBandits)
        ]
