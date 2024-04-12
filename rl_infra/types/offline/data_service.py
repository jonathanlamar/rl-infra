from typing import Protocol, Sequence, TypeVar

from rl_infra.types.offline.schema import OnlineMetrics
from rl_infra.types.online.environment import EpisodeRecord, GameplayRecord
from rl_infra.types.online.transition import Action, State, Transition

A = TypeVar("A", bound=Action)
OM = TypeVar("OM", bound=OnlineMetrics)
S = TypeVar("S", bound=State)


class DataService(Protocol[S, A, OM]):
    capacity: int

    def pushEpisode(self, episode: EpisodeRecord[S, A, OM]) -> None: ...

    def pushValidationEpisode(self, episode: EpisodeRecord[S, A, OM]) -> None: ...

    def getValidationEpisode(self, episodeId: int | None = None) -> EpisodeRecord[S, A, OM]: ...

    def pushGameplay(self, gameplay: GameplayRecord[S, A, OM]) -> None: ...

    def sample(self, batchSize: int) -> Sequence[Transition[S, A]]: ...

    def keepNewRowsDeleteOld(self, sgn: int) -> None: ...
