from typing import Protocol

from online.types.environment import StepOutcome
from online.types.space import Action, State


class Policy(Protocol):
    def onlineUpdate(self, state: State, action: Action, outcome: StepOutcome) -> None:
        raise NotImplementedError
