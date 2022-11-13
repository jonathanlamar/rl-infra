from dataclasses import dataclass
from typing import Optional, Protocol

from online.types.space import Action, ActionSpace, State, StateSpace


@dataclass
class StepOutcome:
    newState: State
    reward: float
    isTerminal: bool


class Environment(Protocol):
    actionSpace: ActionSpace
    stateSpace: StateSpace
    currentState: State

    def step(self, action: Action) -> StepOutcome:
        raise NotImplementedError

    def reset(self, seed: Optional[int], *args, **kwargs) -> State:
        raise NotImplementedError
