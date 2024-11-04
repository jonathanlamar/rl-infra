from typing import Generic, Type, TypeVar

from rl_infra.nonstationary.context import NonstationaryContext
from rl_infra.nonstationary.environment import NonstationaryBanditEnvironment
from rl_infra.types.agent import Agent
from rl_infra.types.bandit_problem import BanditProblem
from rl_infra.types.history import History
from rl_infra.types.transition import Action

A = TypeVar("A", bound=Agent)


class NonstationaryBanditProblem(
    BanditProblem[NonstationaryContext, Action, A], Generic[A]
):
    r"""
    Stationary bandit problem.  It operates in the stationary setting, which is context
    free.  As such, its agent must be a subclass of StationaryBanditAgent.
    """

    agent: A
    environment: NonstationaryBanditEnvironment
    history: History[NonstationaryContext, Action]

    def __init__(self, agentClass: Type[A], **kwargs) -> None:
        self.agent = agentClass(**kwargs)
        self.environment = NonstationaryBanditEnvironment()  # pyright: ignore
        self.history = History[NonstationaryContext, Action]()
