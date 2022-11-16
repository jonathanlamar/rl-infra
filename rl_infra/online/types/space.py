from typing import Protocol, TypeVar

# Space is a parameterized type.  Apparently this is how you do type
# parameterization in python.  Lol
T = TypeVar("T", covariant=False, contravariant=False)


# States will vary quite a bit between implementations, so I am just using this
# class as a type stub.
class State:
    pass


# Action spaces should always be countable.  Maybe we can change this in the
# future, but I will be using int-typed enums for action spaces.
Action = int


class Space(Protocol[T]):
    def sample(self) -> T:
        raise NotImplementedError

    def contains(self, element: T) -> bool:
        # Not sure if this needs to exist in all circumstances.
        # Maybe should be a method implemented only for gym spaces.
        raise NotImplementedError


# These may not be necessary if they do not contain extra methods.
class StateSpace(Space[State], Protocol):
    pass


class ActionSpace(Space[Action], Protocol):
    pass
