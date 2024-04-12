from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

from typing_extensions import Self

from rl_infra.types.base_types import SerializableDataClass
from rl_infra.types.offline.schema import OnlineMetrics
from rl_infra.types.online.transition import Action, State, Transition

S_co = TypeVar("S_co", bound=State, covariant=True)
A_co = TypeVar("A_co", bound=Action, covariant=True)
OM_co = TypeVar("OM_co", bound=OnlineMetrics, covariant=True)


class EpisodeRecord(ABC, SerializableDataClass, Generic[S_co, A_co, OM_co]):
    episodeNumber: int
    moves: list[Transition[S_co, A_co]]

    @abstractmethod
    def computeOnlineMetrics(self) -> OM_co: ...

    def append(self, transition: Transition[S_co, A_co]) -> Self:
        return self.__class__(episodeNumber=self.episodeNumber, moves=self.moves + [transition])


class GameplayRecord(SerializableDataClass, Generic[S_co, A_co, OM_co]):
    episodes: list[EpisodeRecord[S_co, A_co, OM_co]]

    def appendEpisode(self, episode: EpisodeRecord[S_co, A_co, OM_co]) -> Self:
        return self.__class__(episodes=self.episodes + [episode])


S = TypeVar("S", bound=State)
A = TypeVar("A", bound=Action)
OM = TypeVar("OM", bound=OnlineMetrics)


class Environment(Protocol[S, A, OM]):
    currentState: S
    currentEpisodeRecord: EpisodeRecord[S, A, OM]
    currentGameplayRecord: GameplayRecord[S, A, OM]

    def step(self, action: A) -> Transition[S, A]: ...

    def getReward(self, oldState: S, action: A, newState: S) -> float: ...

    def startNewEpisode(self) -> None: ...
