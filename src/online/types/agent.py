from typing import Protocol

from online.types.space import ActionSpace, State, Action, StateSpace
from online.types.environment import StepOutcome


class Agent(Protocol):
    stateSpace: StateSpace
    ActionSpace: ActionSpace

    def act(self, state: State) -> Action:
        raise NotImplementedError

    def updatePolicy(self, state: State, action: Action, outcome: StepOutcome) -> None:
        raise NotImplementedError
