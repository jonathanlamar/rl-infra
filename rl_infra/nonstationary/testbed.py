from typing import Generic, Sequence, Type, TypeVar

from rl_infra.nonstationary.bandit_problem import NonstationaryBanditProblem
from rl_infra.types.agent import Agent
from rl_infra.types.testbed import TestBed
from rl_infra.types.transition import Action, Context

Ag = TypeVar("Ag", bound=Agent)


class NonstationaryBanditTestBed(TestBed[Context, Action, Ag], Generic[Ag]):
    r"""
    A testbed of nonstationary bandits.  This is a list of nonstationary bandit problems
    along with a method for making all of them play.
    """

    bandits: Sequence[NonstationaryBanditProblem[Ag]]

    def __init__(self, agentClass: Type[Ag], numBandits: int, **kwargs) -> None:
        self.bandits = [  # pyright: ignore
            NonstationaryBanditProblem(agentClass, **kwargs) for _ in range(numBandits)
        ]
