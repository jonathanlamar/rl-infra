from typing import Generic, Type, TypeVar

from rl_infra.stationary.agent import StationaryBanditAgent
from rl_infra.stationary.environment import StationaryBanditEnvironment
from rl_infra.types.bandit_problem import BanditProblem
from rl_infra.types.history import History
from rl_infra.types.transition import Action, Context

A = TypeVar("A", bound=StationaryBanditAgent)


class StationaryBanditProblem(BanditProblem[Context, Action, A], Generic[A]):
    r"""
    Stationary bandit problem.  It operates in the stationary setting, which is context
    free.  As such, its agent must be a subclass of StationaryBanditAgent.
    """

    agent: A
    environment: StationaryBanditEnvironment
    history: History[Context, Action]

    def __init__(self, agentClass: Type[A], **kwargs) -> None:
        self.agent = agentClass(**kwargs)
        self.environment = StationaryBanditEnvironment(numArms=kwargs["numArms"])  # pyright: ignore
        self.history = History[Context, Action]()
