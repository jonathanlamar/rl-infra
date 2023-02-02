import random
from collections import deque
from typing import Deque, List, Protocol, TypeVar

from rl_infra.base_types import SerializableDataClass
from rl_infra.online.impl.tetris.tetris_environment import TetrisTransition
from rl_infra.online.types.environment import Transition

T = TypeVar("T", bound=Transition, covariant=False, contravariant=False)


class ReplayMemory(Protocol[T]):
    memory: Deque[T]

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, transition: T) -> None:
        self.memory.append(transition)

    def sample(self, batchSize: int) -> List[T]:
        return random.sample(self.memory, batchSize)

    def __len__(self) -> int:
        return len(self.memory)


class DataDbEntry(SerializableDataClass):
    transition: TetrisTransition
