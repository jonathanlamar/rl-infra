from typing import Protocol, TypeVar

# Space is a parameterized type.  Apparently this is how you do type
# parameterization in python.  Lol
T = TypeVar("T", covariant=False, contravariant=False)


class State:
    pass


class Action:
    pass


class Space(Protocol[T]):
    def sample(self) -> T:
        raise NotImplementedError

    def contains(self, element: T) -> bool:
        raise NotImplementedError


# These may not be necessary if they do not contain extra methods.
class StateSpace(Space[State], Protocol):
    pass


class ActionSpace(Space[Action], Protocol):
    pass
